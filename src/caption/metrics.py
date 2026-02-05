"""
Caption metrics tracking module.

Provides CaptionMetrics class for tracking caption fetch statistics
with thread-safe operations for parallel fetching.

Extracted from caption_fetcher.py for US-34-004.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .enums import CaptionErrorCategory

if TYPE_CHECKING:
    from .cache_models import CacheValidationResult, ChannelCaptionPattern

logger = logging.getLogger(__name__)


@dataclass
class CaptionMetrics:
    """
    Tracks caption fetch statistics for pipeline reporting.

    Implements US-011: Add caption fetch metrics and reporting.
    Updated US-001: Thread-safe with Lock for parallel caption fetching.
    Updated US-002: Tracks skipped live streams separately.
    Updated US-004: Tracks coverage distribution (high/medium/low).
    Updated US-008: Tracks pre-check availability results.
    Updated US-003 Sprint 6: Language selection audit trail for fallback debugging.
    Updated US-004 Sprint 6: Format preference success rate telemetry.
    Updated US-003 Sprint 7: Error category tracking for type-specific retry debugging.
    Updated US-34-004: Extracted to dedicated module with convenience methods.

    Tracks:
    - Fetch attempts, successes, failures, cache hits
    - Skipped live streams (US-002)
    - Pre-check results: available/unavailable counts (US-008)
    - Language distribution
    - Quality distribution (human, auto, unavailable)
    - Coverage distribution (high >80%, medium 50-80%, low <50%) (US-004)
    - Total segments fetched
    - Language selection audit trail (US-003 Sprint 6)
    - Format success counts and fallback tracking (US-004 Sprint 6)
    - Error category counts (US-003 Sprint 7)

    Thread Safety:
        All mutation methods are protected by a Lock for concurrent access
        during parallel caption fetching with ThreadPoolExecutor.

    Usage:
        metrics = CaptionMetrics()
        metrics.record_fetch_attempt("dQw4w9WgXcQ")
        metrics.record_fetch_success("dQw4w9WgXcQ", language="en", quality="high", coverage_ratio=0.85)

        # Convenience methods (US-34-004):
        metrics.increment_success()
        metrics.increment_failure()
        metrics.increment_cached()

        # Get summary dict:
        summary = metrics.get_summary_dict()
        print(f"Hit rate: {summary['hit_rate']}%")
        print(f"Success rate: {summary['success_rate']}%")

        # Full summary:
        print(metrics.summary())

    Attributes:
        fetch_attempts: Total fetch attempts made
        successes: Number of successful caption fetches
        failures: Number of failed caption fetches (unavailable/error)
        cache_hits: Number of cache hits (captions loaded from cache)
        skipped_live_streams: Number of live streams skipped (US-002)
        pre_check_available: Number of videos with captions (pre-check) (US-008)
        pre_check_unavailable: Number of videos without captions (pre-check) (US-008)
        language_distribution: Dict mapping language code -> count
        quality_distribution: Dict mapping quality level -> count
        coverage_distribution: Dict mapping coverage level -> count (US-004)
        language_selection_trace: List of language selection audit entries (US-003 Sprint 6)
        format_success_counts: Dict mapping format -> success count (US-004 Sprint 6)
        format_fallback_count: Videos needing format != first preference (US-004 Sprint 6)
        video_format_used: Dict mapping video_id -> format used (US-004 Sprint 6)
        error_category_counts: Dict mapping error category name -> count (US-003 Sprint 7)
        total_segments: Total caption segments fetched
    """

    # Core counts
    fetch_attempts: int = 0
    successes: int = 0
    failures: int = 0
    cache_hits: int = 0

    # Skipped live streams (US-002)
    skipped_live_streams: int = 0

    # Pre-check results (US-008)
    pre_check_available: int = 0
    pre_check_unavailable: int = 0

    # Batch pre-check tracking (US-006 Sprint 7)
    # Tracks API calls saved through channel-based batch pre-checking
    pre_check_batched_total: int = 0  # Total videos in batch pre-check
    pre_check_batched_skipped: int = 0  # Videos skipped due to channel pattern
    pre_check_batched_checked: int = 0  # Videos actually checked
    pre_check_api_calls_saved: int = 0  # API calls saved vs individual checks

    # Distribution tracking
    language_distribution: Dict[str, int] = field(default_factory=dict)
    quality_distribution: Dict[str, int] = field(default_factory=dict)

    # Coverage distribution (US-004): high (>80%), medium (50-80%), low (<50%)
    coverage_distribution: Dict[str, int] = field(default_factory=dict)

    # Low coverage video tracking (US-004)
    low_coverage_videos: List[str] = field(default_factory=list)

    # Additional metrics
    total_segments: int = 0
    auto_generated_count: int = 0
    human_caption_count: int = 0

    # Per-video timing tracking (US-002 Sprint 6)
    # Dict mapping video_id to elapsed_seconds for slowest videos analysis
    video_fetch_times: Dict[str, float] = field(default_factory=dict)

    # Language selection audit trail (US-003 Sprint 6)
    # Each entry: {video_id, attempted_codes, selected_code, selection_reason, is_auto_generated}
    language_selection_trace: List[Dict[str, Any]] = field(default_factory=list)

    # Format preference tracking (US-004 Sprint 6)
    # Dict mapping format name -> success count (e.g., {'json3': 92, 'vtt': 8})
    format_success_counts: Dict[str, int] = field(default_factory=dict)
    # Count of videos that needed format != first preference
    format_fallback_count: int = 0
    # Dict mapping video_id -> format that succeeded for that video
    video_format_used: Dict[str, str] = field(default_factory=dict)

    # Cache validation results (US-008 Sprint 6)
    # Tracks: {passed: N, rejected: M, refetched: K}
    cache_validation_passed: int = 0
    cache_validation_rejected: int = 0
    cache_validation_refetched: int = 0

    # Error category tracking (US-003 Sprint 7)
    # Dict mapping error category name -> count (e.g., {'NETWORK': 5, 'PARSE': 2})
    # Used for debugging retry behavior and identifying error patterns
    error_category_counts: Dict[str, int] = field(default_factory=dict)

    # Channel-level caption availability patterns (US-009 Sprint 7)
    # Dict mapping channel_id -> ChannelCaptionPattern for cross-project learning
    channel_patterns: Dict[str, 'ChannelCaptionPattern'] = field(default_factory=dict)

    # Batch retry budget tracking (US-001 Sprint 8)
    # Summary dict stored here after batch completion
    batch_retry_budget: Optional[Dict[str, Any]] = None

    # US-59-006: Batch unavailable tracking
    # Count and rate of CaptionUnavailableError results in the batch
    unavailable_count: int = 0
    unavailable_rate: float = 0.0

    # Per-format attempt tracking (US-59-009)
    # Each entry: {video_id, format_name, success, elapsed_seconds}
    format_attempt_records: List[Dict[str, Any]] = field(default_factory=list)
    # Count of subprocess calls avoided by pre-flight list-subs check (US-59-009)
    calls_saved_by_preflight: int = 0
    # Count of subprocess calls avoided by negative cache hit (US-59-009)
    calls_saved_by_negative_cache: int = 0

    # US-62-005: Auto-generated fallback tracking
    # Count of videos where manual captions failed and auto-generated was used as fallback
    auto_fallback_count: int = 0

    # Thread-safety lock (US-001) - not serialized
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    # =========================================================================
    # Convenience Methods (US-34-004)
    # =========================================================================

    def increment_success(self) -> None:
        """Thread-safe increment of success counter.

        Simple convenience method for incrementing successes without
        full record_fetch_success() call.

        Thread-safe: Protected by lock for parallel fetching.
        """
        with self._lock:
            self.successes += 1

    def increment_failure(self) -> None:
        """Thread-safe increment of failure counter.

        Simple convenience method for incrementing failures without
        full record_fetch_failure() call.

        Thread-safe: Protected by lock for parallel fetching.
        """
        with self._lock:
            self.failures += 1

    def increment_cached(self) -> None:
        """Thread-safe increment of cache hit counter.

        Simple convenience method for incrementing cache_hits without
        full record_cache_hit() call.

        Thread-safe: Protected by lock for parallel fetching.
        """
        with self._lock:
            self.cache_hits += 1

    def get_summary_dict(self) -> Dict[str, Any]:
        """Get summary statistics as a dictionary.

        Returns hit rate, success rate, and average fetch time for
        quick access to key metrics.

        Thread-safe: Protected by lock for parallel fetching.

        Returns:
            Dict with summary statistics:
            - hit_rate: Cache hit rate percentage (0-100)
            - success_rate: Fetch success rate percentage (0-100)
            - avg_fetch_time: Average fetch time in seconds (or 0.0 if no timing data)
            - total_processed: Total videos processed
            - successes: Number of successful fetches
            - failures: Number of failed fetches
            - cache_hits: Number of cache hits
            - total_segments: Total caption segments fetched
        """
        with self._lock:
            total_processed = self.successes + self.failures + self.cache_hits + self.skipped_live_streams

            # Calculate hit rate
            hit_rate = 0.0
            if total_processed > 0:
                hit_rate = round(100.0 * self.cache_hits / total_processed, 1)

            # Calculate success rate
            fetch_total = self.successes + self.failures
            success_rate = 0.0
            if fetch_total > 0:
                success_rate = round(100.0 * self.successes / fetch_total, 1)

            # Calculate average fetch time
            avg_fetch_time = 0.0
            if self.video_fetch_times:
                avg_fetch_time = round(
                    sum(self.video_fetch_times.values()) / len(self.video_fetch_times),
                    3
                )

            return {
                'hit_rate': hit_rate,
                'success_rate': success_rate,
                'avg_fetch_time': avg_fetch_time,
                'total_processed': total_processed,
                'successes': self.successes,
                'failures': self.failures,
                'cache_hits': self.cache_hits,
                'total_segments': self.total_segments,
                'unavailable_count': self.unavailable_count,
                'unavailable_rate': self.unavailable_rate,
            }

    # =========================================================================
    # Recording Methods
    # =========================================================================

    def record_fetch_attempt(self, video_id: str = "") -> None:
        """Record a caption fetch attempt.

        Thread-safe: Protected by lock for parallel fetching.

        Args:
            video_id: Optional video ID for logging context.
        """
        with self._lock:
            self.fetch_attempts += 1
        logger.debug(f"Caption fetch attempt recorded for {video_id or 'unknown'}")

    def record_format_attempt(
        self,
        video_id: str,
        format_name: str,
        success: bool,
        elapsed_seconds: float
    ) -> None:
        """Record a per-format subprocess attempt with timing (US-59-009).

        Thread-safe: Protected by lock for parallel fetching.

        Args:
            video_id: YouTube video ID.
            format_name: Caption format attempted (e.g., 'json3', 'vtt', 'srt').
            success: Whether the subprocess call succeeded.
            elapsed_seconds: Wall-clock time for the subprocess call.
        """
        with self._lock:
            self.format_attempt_records.append({
                'video_id': video_id,
                'format_name': format_name,
                'success': success,
                'elapsed_seconds': elapsed_seconds,
            })

        logger.debug(
            f"Format attempt recorded for {video_id}: {format_name} "
            f"{'OK' if success else 'FAIL'} in {elapsed_seconds:.2f}s"
        )

    def record_preflight_saving(self, video_id: str = "") -> None:
        """Record a subprocess call saved by pre-flight list-subs check (US-59-009).

        Thread-safe: Protected by lock for parallel fetching.

        Args:
            video_id: Video ID for logging context.
        """
        with self._lock:
            self.calls_saved_by_preflight += 1

        logger.debug(f"Pre-flight saved subprocess call for {video_id or 'unknown'}")

    def record_negative_cache_saving(self, video_id: str = "") -> None:
        """Record a subprocess call saved by negative cache hit (US-59-009).

        Thread-safe: Protected by lock for parallel fetching.

        Args:
            video_id: Video ID for logging context.
        """
        with self._lock:
            self.calls_saved_by_negative_cache += 1

        logger.debug(f"Negative cache saved subprocess call for {video_id or 'unknown'}")

    def record_auto_fallback(self, video_id: str, language: str = "en") -> None:
        """Record auto-generated fallback usage when manual captions unavailable (US-62-005).

        Thread-safe: Protected by lock for parallel fetching.

        Called when manual caption fetch fails and auto-generated captions are
        used as a fallback. This is separate from auto_generated_count which
        tracks all auto-generated captions regardless of fallback.

        Args:
            video_id: YouTube video ID.
            language: Language code that was fetched.
        """
        with self._lock:
            self.auto_fallback_count += 1

        logger.info(
            f"Auto-generated fallback used for {video_id}: "
            f"selection_reason='auto-generated fallback'"
        )

        # Also record in language selection trace for audit trail
        self.record_language_selection(
            video_id=video_id,
            attempted_codes=[language],
            selected_code=language,
            selection_reason='auto-generated fallback',
            is_auto_generated=True
        )

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get caption fetch performance summary (US-59-009).

        Thread-safe: Protected by lock for parallel fetching.

        Returns:
            Dict with performance metrics:
            - total_subprocess_calls: Total subprocess invocations recorded
            - total_subprocess_seconds: Cumulative wall-clock time in subprocess calls
            - avg_seconds_per_call: Average duration per subprocess call
            - calls_saved_by_preflight: Calls avoided by pre-flight list-subs check
            - calls_saved_by_negative_cache: Calls avoided by negative cache hits
        """
        with self._lock:
            total_calls = len(self.format_attempt_records)
            total_seconds = sum(
                r['elapsed_seconds'] for r in self.format_attempt_records
            )
            avg_seconds = total_seconds / total_calls if total_calls > 0 else 0.0

            return {
                'total_subprocess_calls': total_calls,
                'total_subprocess_seconds': round(total_seconds, 2),
                'avg_seconds_per_call': round(avg_seconds, 2),
                'calls_saved_by_preflight': self.calls_saved_by_preflight,
                'calls_saved_by_negative_cache': self.calls_saved_by_negative_cache,
                'error_counts': dict(self.error_category_counts),  # US-61-007
            }

    def record_fetch_success(
        self,
        video_id: str = "",
        language: str = "en",
        quality: str = "medium",
        segment_count: int = 0,
        is_auto_generated: bool = False,
        coverage_ratio: Optional[float] = None,
        min_coverage_threshold: float = 0.5,
        elapsed_seconds: Optional[float] = None,
        format_source: Optional[str] = None,
        preferred_format: Optional[str] = None
    ) -> None:
        """Record a successful caption fetch.

        Thread-safe: Protected by lock for parallel fetching.

        Args:
            video_id: Video ID for logging.
            language: ISO 639-1 language code (e.g., 'en').
            quality: Caption quality level ('high', 'medium', 'low').
            segment_count: Number of caption segments fetched.
            is_auto_generated: Whether captions are auto-generated.
            coverage_ratio: Caption coverage ratio 0.0-1.0 (US-004).
            min_coverage_threshold: Threshold for low coverage warning (US-004).
            elapsed_seconds: Time taken for this fetch in seconds (US-002 Sprint 6).
            format_source: Caption format used (e.g., 'json3', 'vtt', 'srt') (US-004 Sprint 6).
            preferred_format: First format in preference list (US-004 Sprint 6).
        """
        with self._lock:
            self.successes += 1
            self.total_segments += segment_count

            # Track language distribution
            self.language_distribution[language] = self.language_distribution.get(language, 0) + 1

            # Track quality distribution
            self.quality_distribution[quality] = self.quality_distribution.get(quality, 0) + 1

            # Track auto vs human
            if is_auto_generated:
                self.auto_generated_count += 1
            else:
                self.human_caption_count += 1

            # Track coverage distribution (US-004)
            if coverage_ratio is not None:
                coverage_level = self._classify_coverage(coverage_ratio)
                self.coverage_distribution[coverage_level] = (
                    self.coverage_distribution.get(coverage_level, 0) + 1
                )
                # Track low coverage videos for warning
                if coverage_ratio < min_coverage_threshold and video_id:
                    self.low_coverage_videos.append(video_id)

            # Track per-video fetch time (US-002 Sprint 6)
            if elapsed_seconds is not None and video_id:
                self.video_fetch_times[video_id] = elapsed_seconds

            # Track format success (US-004 Sprint 6)
            if format_source:
                self.format_success_counts[format_source] = (
                    self.format_success_counts.get(format_source, 0) + 1
                )
                if video_id:
                    self.video_format_used[video_id] = format_source
                # Track fallback: format != preferred_format
                if preferred_format and format_source != preferred_format:
                    self.format_fallback_count += 1

        logger.debug(
            f"Caption fetch success for {video_id or 'unknown'}: "
            f"lang={language}, quality={quality}, segments={segment_count}, "
            f"coverage={coverage_ratio:.1%}" if coverage_ratio else
            f"Caption fetch success for {video_id or 'unknown'}: "
            f"lang={language}, quality={quality}, segments={segment_count}"
        )

    @staticmethod
    def _classify_coverage(coverage_ratio: float) -> str:
        """Classify coverage ratio into high/medium/low category (US-004).

        Args:
            coverage_ratio: Coverage ratio from 0.0 to 1.0.

        Returns:
            'high' if >80%, 'medium' if 50-80%, 'low' if <50%.
        """
        if coverage_ratio > 0.8:
            return 'high'
        elif coverage_ratio >= 0.5:
            return 'medium'
        else:
            return 'low'

    def record_fetch_failure(
        self,
        video_id: str = "",
        reason: str = "unknown",
        elapsed_seconds: Optional[float] = None
    ) -> None:
        """Record a failed caption fetch.

        Thread-safe: Protected by lock for parallel fetching.

        Args:
            video_id: Video ID for logging.
            reason: Reason for failure (e.g., 'unavailable', 'error', 'timeout').
            elapsed_seconds: Time taken before failure in seconds (US-002 Sprint 6).
        """
        with self._lock:
            self.failures += 1

            # Track as 'unavailable' quality
            self.quality_distribution['unavailable'] = self.quality_distribution.get('unavailable', 0) + 1

            # Track per-video fetch time even for failures (US-002 Sprint 6)
            if elapsed_seconds is not None and video_id:
                self.video_fetch_times[video_id] = elapsed_seconds

        logger.debug(f"Caption fetch failure for {video_id or 'unknown'}: {reason}")

    def record_unavailable_summary(
        self,
        count: int,
        rate: float,
    ) -> None:
        """Record batch-level unavailable summary (US-59-006).

        Called by BatchProcessor after batch completion to store the
        total unavailable count and rate in the batch summary.

        Thread-safe: Protected by lock for parallel fetching.

        Args:
            count: Total number of CaptionUnavailableError results.
            rate: Ratio of unavailable to total fetched (0.0-1.0).
        """
        with self._lock:
            self.unavailable_count = count
            self.unavailable_rate = round(rate, 4)

    def record_error_category(
        self,
        category: 'CaptionErrorCategory',
        video_id: str = ""
    ) -> None:
        """Record an error category occurrence (US-003 Sprint 7).

        Thread-safe: Protected by lock for parallel fetching.

        Tracks the distribution of error types encountered during caption
        fetching. This data is used for debugging retry strategies and
        identifying systemic issues (e.g., high rate of PARSE errors
        indicates bad caption sources).

        Args:
            category: The CaptionErrorCategory enum value.
            video_id: Optional video ID for logging context.

        Example:
            >>> metrics.record_error_category(CaptionErrorCategory.NETWORK, "abc123")
            >>> metrics.record_error_category(CaptionErrorCategory.PARSE, "def456")
        """
        category_name = category.name
        with self._lock:
            self.error_category_counts[category_name] = (
                self.error_category_counts.get(category_name, 0) + 1
            )

        logger.debug(f"Error category recorded for {video_id or 'unknown'}: {category_name}")

    def set_batch_retry_budget(self, budget_summary: Dict[str, Any]) -> None:
        """Store batch retry budget summary (US-001 Sprint 8).

        Thread-safe: Protected by lock for parallel fetching.

        Args:
            budget_summary: Dict from BatchRetryBudget.get_summary() containing:
                - 'total_videos': Total videos in batch
                - 'processed_videos': Videos processed
                - 'error_rates': Dict mapping category -> rate
                - 'original_budgets': Original per-category budgets
                - 'reduced_budgets': Final per-category budgets (may be reduced)
                - 'reductions_applied': Number of threshold reductions triggered
                - 'estimated_retries_saved': Estimated retries avoided
        """
        with self._lock:
            self.batch_retry_budget = budget_summary

        reductions = budget_summary.get('reductions_applied', 0)
        saved = budget_summary.get('estimated_retries_saved', 0)
        logger.debug(
            f"Batch retry budget recorded: {reductions} reductions, "
            f"~{saved} retries saved"
        )

    def get_error_category_summary(self) -> Dict[str, Any]:
        """Get summary of error categories (US-003 Sprint 7).

        Thread-safe: Protected by lock for parallel fetching.

        Returns:
            Dict with error category statistics:
            - 'counts': Dict mapping category name -> count
            - 'total': Total errors categorized
            - 'top_category': Most frequent error category
            - 'category_rates': Dict mapping category -> percentage
        """
        with self._lock:
            counts = dict(self.error_category_counts)
            total = sum(counts.values())

            if total == 0:
                return {
                    'counts': {},
                    'total': 0,
                    'top_category': None,
                    'category_rates': {}
                }

            # Calculate rates
            category_rates = {}
            for cat, count in counts.items():
                category_rates[cat] = round(100.0 * count / total, 1)

            # Find top category
            top_category = max(counts, key=counts.get) if counts else None

            return {
                'counts': counts,
                'total': total,
                'top_category': top_category,
                'category_rates': category_rates
            }

    def record_error_pattern_detected(
        self,
        error_signature: str,
        affected_count: int,
        sample_size: int
    ) -> None:
        """Record that an error pattern was detected (US-007 Sprint 7).

        Thread-safe: Protected by lock for parallel fetching.

        Args:
            error_signature: The canonical error signature (e.g., "403 Forbidden").
            affected_count: Number of videos affected by this error.
            sample_size: Total videos checked when pattern was detected.
        """
        with self._lock:
            if not hasattr(self, 'error_patterns_detected'):
                self.error_patterns_detected: List[Dict[str, Any]] = []
            self.error_patterns_detected.append({
                'error_signature': error_signature,
                'affected_count': affected_count,
                'sample_size': sample_size,
                'ratio': affected_count / sample_size if sample_size > 0 else 0,
            })

        logger.debug(
            f"Error pattern recorded: {error_signature} "
            f"({affected_count}/{sample_size} videos)"
        )

    def set_channel_patterns(
        self,
        patterns: Dict[str, 'ChannelCaptionPattern']
    ) -> None:
        """Set channel-level caption availability patterns (US-009 Sprint 7).

        Thread-safe: Protected by lock for parallel fetching.

        Args:
            patterns: Dict mapping channel_id -> ChannelCaptionPattern.
        """
        with self._lock:
            self.channel_patterns = dict(patterns)

        logger.debug(f"Channel patterns set: {len(patterns)} channels")

    def get_channel_statistics(self, top_n: int = 5) -> Dict[str, Any]:
        """Get channel-level caption availability statistics (US-009 Sprint 7).

        Thread-safe: Protected by lock for parallel fetching.

        Args:
            top_n: Number of top/bottom channels to return (default: 5).

        Returns:
            Dict with channel statistics:
            - 'total_channels': Total number of channels tracked
            - 'total_videos_checked': Sum of videos_checked across all channels
            - 'top_channels': List of (channel_id, success_rate, videos_checked) tuples
            - 'bottom_channels': List of (channel_id, success_rate, videos_checked) tuples
            - 'avg_success_rate': Average success rate across all channels
            - 'channels_with_100pct': Count of channels with 100% caption availability
            - 'channels_with_0pct': Count of channels with 0% caption availability
        """
        with self._lock:
            if not self.channel_patterns:
                return {
                    'total_channels': 0,
                    'total_videos_checked': 0,
                    'top_channels': [],
                    'bottom_channels': [],
                    'avg_success_rate': 0.0,
                    'channels_with_100pct': 0,
                    'channels_with_0pct': 0,
                }

            # Build list of (channel_id, success_rate, videos_checked)
            channel_data = [
                (cid, pattern.success_rate, pattern.videos_checked)
                for cid, pattern in self.channel_patterns.items()
                if pattern.videos_checked > 0
            ]

            if not channel_data:
                return {
                    'total_channels': len(self.channel_patterns),
                    'total_videos_checked': 0,
                    'top_channels': [],
                    'bottom_channels': [],
                    'avg_success_rate': 0.0,
                    'channels_with_100pct': 0,
                    'channels_with_0pct': 0,
                }

            # Sort by success rate descending for top, ascending for bottom
            sorted_by_success = sorted(channel_data, key=lambda x: (-x[1], -x[2]))
            top_channels = sorted_by_success[:top_n]
            bottom_channels = sorted(channel_data, key=lambda x: (x[1], -x[2]))[:top_n]

            # Calculate statistics
            total_videos = sum(c[2] for c in channel_data)
            avg_success = sum(c[1] for c in channel_data) / len(channel_data) if channel_data else 0.0
            perfect_channels = sum(1 for c in channel_data if c[1] >= 1.0)
            zero_channels = sum(1 for c in channel_data if c[1] <= 0.0)

            return {
                'total_channels': len(self.channel_patterns),
                'total_videos_checked': total_videos,
                'top_channels': top_channels,
                'bottom_channels': bottom_channels,
                'avg_success_rate': round(avg_success, 3),
                'channels_with_100pct': perfect_channels,
                'channels_with_0pct': zero_channels,
            }

    def get_channel_summary(self) -> str:
        """Get human-readable channel availability summary (US-009 Sprint 7).

        Returns a formatted string suitable for printing in pipeline reports.
        """
        stats = self.get_channel_statistics(top_n=3)

        if stats['total_channels'] == 0:
            return ""

        parts = []

        # Format top channels
        if stats['top_channels']:
            top_strs = [
                f"{cid[:10]}{'...' if len(cid) > 10 else ''} ({rate:.0%})"
                for cid, rate, _ in stats['top_channels'][:3]
            ]
            parts.append(f"Top channels: {', '.join(top_strs)}")

        # Format bottom channels (only if different from top and have issues)
        bottom_with_issues = [
            (cid, rate, count) for cid, rate, count in stats['bottom_channels']
            if rate < 0.5  # Only show channels with <50% success
        ]
        if bottom_with_issues:
            bottom_strs = [
                f"{cid[:10]}{'...' if len(cid) > 10 else ''} ({rate:.0%})"
                for cid, rate, _ in bottom_with_issues[:3]
            ]
            parts.append(f"Bottom: {', '.join(bottom_strs)}")

        return ". ".join(parts)

    def record_skipped_live_stream(self, video_id: str = "") -> None:
        """Record a skipped live stream (US-002).

        Thread-safe: Protected by lock for parallel fetching.

        Args:
            video_id: Video ID for logging.
        """
        with self._lock:
            self.skipped_live_streams += 1

            # Track as 'skipped' quality for distribution
            self.quality_distribution['skipped'] = self.quality_distribution.get('skipped', 0) + 1

        logger.debug(f"Caption fetch skipped for live stream: {video_id or 'unknown'}")

    def record_pre_check(self, video_id: str, has_captions: bool) -> None:
        """Record a caption availability pre-check result (US-008).

        Thread-safe: Protected by lock for parallel fetching.

        Args:
            video_id: Video ID that was checked.
            has_captions: True if captions are available, False if unavailable.
        """
        with self._lock:
            if has_captions:
                self.pre_check_available += 1
            else:
                self.pre_check_unavailable += 1

        status = "available" if has_captions else "unavailable"
        logger.debug(f"Caption pre-check for {video_id}: {status}")

    def record_pre_check_batched(
        self,
        video_id: str,
        skipped: bool
    ) -> None:
        """Record a batch pre-check result (US-006 Sprint 7).

        Thread-safe: Protected by lock for parallel fetching.

        Args:
            video_id: Video ID that was processed.
            skipped: True if pre-check was skipped (used channel pattern),
                    False if individual API call was made.
        """
        with self._lock:
            self.pre_check_batched_total += 1
            if skipped:
                self.pre_check_batched_skipped += 1
            else:
                self.pre_check_batched_checked += 1

        logger.debug(
            f"Batch pre-check for {video_id}: {'skipped' if skipped else 'checked'}"
        )

    def set_batch_precheck_savings(self, api_calls_saved: int) -> None:
        """Set the total API calls saved by batch pre-check (US-006 Sprint 7).

        Thread-safe: Protected by lock for parallel fetching.

        Args:
            api_calls_saved: Number of API calls saved.
        """
        with self._lock:
            self.pre_check_api_calls_saved = api_calls_saved

        logger.debug(f"Batch pre-check saved {api_calls_saved} API calls")

    def record_cache_hit(
        self,
        video_id: str = "",
        language: str = "en",
        quality: str = "medium",
        segment_count: int = 0,
        is_auto_generated: bool = False,
        coverage_ratio: Optional[float] = None,
        min_coverage_threshold: float = 0.5
    ) -> None:
        """Record a cache hit (captions loaded from cache).

        Thread-safe: Protected by lock for parallel fetching.

        Args:
            video_id: Video ID for logging.
            language: ISO 639-1 language code.
            quality: Caption quality level.
            segment_count: Number of segments in cached captions.
            is_auto_generated: Whether cached captions are auto-generated.
            coverage_ratio: Caption coverage ratio 0.0-1.0 (US-004).
            min_coverage_threshold: Threshold for low coverage warning (US-004).
        """
        with self._lock:
            self.cache_hits += 1
            self.total_segments += segment_count

            # Track language distribution
            self.language_distribution[language] = self.language_distribution.get(language, 0) + 1

            # Track quality distribution
            self.quality_distribution[quality] = self.quality_distribution.get(quality, 0) + 1

            # Track auto vs human
            if is_auto_generated:
                self.auto_generated_count += 1
            else:
                self.human_caption_count += 1

            # Track coverage distribution (US-004)
            if coverage_ratio is not None:
                coverage_level = self._classify_coverage(coverage_ratio)
                self.coverage_distribution[coverage_level] = (
                    self.coverage_distribution.get(coverage_level, 0) + 1
                )
                # Track low coverage videos for warning
                if coverage_ratio < min_coverage_threshold and video_id:
                    self.low_coverage_videos.append(video_id)

        logger.debug(f"Caption cache hit for {video_id or 'unknown'}: lang={language}")

    def record_cache_validation(
        self,
        result: 'CacheValidationResult'
    ) -> None:
        """Record a cache validation result (US-008 Sprint 6).

        Thread-safe: Protected by lock for parallel fetching.

        Args:
            result: CacheValidationResult from CaptionCache.validate_cache_entry().
        """
        with self._lock:
            if result.is_valid:
                self.cache_validation_passed += 1
            else:
                self.cache_validation_refetched += 1

        status = "passed" if result.is_valid else "failed"
        logger.debug(
            f"Cache validation {status} for {result.video_id}_{result.language}: "
            f"{result.reason or 'all checks passed'}"
        )

    def get_cache_validation_stats(self) -> Dict[str, Any]:
        """Get cache validation statistics (US-008 Sprint 6).

        Thread-safe: Protected by lock for parallel fetching.

        Returns:
            Dict with validation statistics:
            - passed: Number of entries that passed validation
            - rejected: Number of entries rejected in strict mode
            - refetched: Number of entries re-fetched after validation failure
            - total: Total validations performed
            - pass_rate: Percentage of validations that passed (0.0-100.0)
        """
        with self._lock:
            total = (
                self.cache_validation_passed +
                self.cache_validation_rejected +
                self.cache_validation_refetched
            )
            return {
                'passed': self.cache_validation_passed,
                'rejected': self.cache_validation_rejected,
                'refetched': self.cache_validation_refetched,
                'total': total,
                'pass_rate': round(
                    100.0 * self.cache_validation_passed / max(total, 1), 1
                ),
            }

    def record_language_selection(
        self,
        video_id: str,
        attempted_codes: List[str],
        selected_code: Optional[str],
        selection_reason: str,
        is_auto_generated: bool = False
    ) -> None:
        """Record a language selection decision for audit trail (US-003 Sprint 6).

        Thread-safe: Protected by lock for parallel fetching.

        Args:
            video_id: YouTube video ID.
            attempted_codes: List of language codes tried in order.
            selected_code: The language code that was selected, or None if no language available.
            selection_reason: Human-readable reason for selection.
            is_auto_generated: Whether the selected caption is auto-generated.
        """
        with self._lock:
            trace_entry = {
                'video_id': video_id,
                'attempted_codes': attempted_codes,
                'selected_code': selected_code,
                'selection_reason': selection_reason,
                'is_auto_generated': is_auto_generated
            }
            self.language_selection_trace.append(trace_entry)

        logger.debug(
            f"Language selection for {video_id}: "
            f"tried={attempted_codes}, selected={selected_code} "
            f"({selection_reason}, auto={is_auto_generated})"
        )

    def get_language_fallback_efficiency(self, preferred_language: str = 'en') -> float:
        """Calculate percentage of videos using preferred language (US-003 Sprint 6).

        Thread-safe: Protected by lock for parallel fetching.

        Args:
            preferred_language: The language code to check as "preferred" (default: 'en').

        Returns:
            Percentage (0.0-100.0) of videos that used the preferred language.
        """
        with self._lock:
            if not self.language_selection_trace:
                return 0.0

            preferred_count = sum(
                1 for entry in self.language_selection_trace
                if entry.get('selected_code', '').lower() == preferred_language.lower()
            )
            total = len(self.language_selection_trace)

            return round(100.0 * preferred_count / total, 1)

    def get_language_fallback_summary(self) -> Dict[str, int]:
        """Get summary of language selection by reason category (US-003 Sprint 6).

        Thread-safe: Protected by lock for parallel fetching.

        Returns:
            Dict mapping selection_reason categories to counts.
        """
        summary: Dict[str, int] = {}

        with self._lock:
            for entry in self.language_selection_trace:
                reason = entry.get('selection_reason', 'unknown')

                # Categorize the reason
                if 'preferred' in reason.lower():
                    key = 'preferred'
                elif 'fallback chain position' in reason.lower():
                    try:
                        pos = reason.split()[-1]
                        key = f'fallback_{pos}'
                    except (IndexError, ValueError):
                        key = 'fallback_other'
                elif 'english fallback' in reason.lower():
                    key = 'english_fallback'
                elif 'any available' in reason.lower():
                    key = 'any_available'
                elif 'none' in reason.lower() or entry.get('selected_code') is None:
                    key = 'none'
                else:
                    key = 'other'

                summary[key] = summary.get(key, 0) + 1

        return summary

    def get_format_statistics(self) -> Dict[str, Any]:
        """Get format preference success rate statistics (US-004 Sprint 6).

        Thread-safe: Protected by lock for parallel fetching.

        Returns:
            Dict with format statistics.
        """
        with self._lock:
            total = sum(self.format_success_counts.values())

            if total == 0:
                return {
                    'format_counts': {},
                    'format_rates': {},
                    'fallback_count': 0,
                    'fallback_rate': 0.0,
                    'total_with_format': 0
                }

            # Calculate success rate per format
            format_rates = {}
            for fmt, count in self.format_success_counts.items():
                format_rates[fmt] = round(100.0 * count / total, 1)

            fallback_rate = round(100.0 * self.format_fallback_count / total, 1) if total else 0.0

            return {
                'format_counts': dict(self.format_success_counts),
                'format_rates': format_rates,
                'fallback_count': self.format_fallback_count,
                'fallback_rate': fallback_rate,
                'total_with_format': total
            }

    def get_optimal_format_order(
        self,
        default_formats: Optional[List[str]] = None
    ) -> List[str]:
        """Get formats sorted by historical success rate (US-002 Sprint 7).

        Args:
            default_formats: Default format order to use if no historical data.

        Returns:
            List of format names sorted by success rate (highest first).
        """
        if default_formats is None:
            default_formats = ["json3", "vtt", "srt"]

        with self._lock:
            total = sum(self.format_success_counts.values())

            if total == 0:
                return list(default_formats)

            sorted_formats = sorted(
                self.format_success_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )

            optimal_order = [fmt for fmt, _ in sorted_formats]

            for fmt in default_formats:
                if fmt not in optimal_order:
                    optimal_order.append(fmt)

            return optimal_order

    @property
    def total_processed(self) -> int:
        """Total videos processed (successes + failures + cache_hits + skipped_live_streams)."""
        return self.successes + self.failures + self.cache_hits + self.skipped_live_streams

    @property
    def success_rate(self) -> float:
        """Calculate fetch success rate as percentage.

        Returns:
            Success rate percentage (0-100), or 0.0 if no attempts.
        """
        total = self.successes + self.failures
        if total == 0:
            return 0.0
        return round(100.0 * self.successes / total, 1)

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate as percentage.

        Returns:
            Cache hit rate percentage (0-100), or 0.0 if no processed videos.
        """
        total = self.total_processed
        if total == 0:
            return 0.0
        return round(100.0 * self.cache_hits / total, 1)

    def summary(self) -> str:
        """Generate human-readable summary for pipeline report.

        Returns:
            Multi-line string suitable for printing in the report.
        """
        lines = []

        # Basic stats
        basic_stats = (
            f"Caption fetch: {self.fetch_attempts} attempts, "
            f"{self.successes} succeeded, {self.failures} failed, "
            f"{self.cache_hits} from cache"
        )
        if self.skipped_live_streams > 0:
            basic_stats += f", {self.skipped_live_streams} live streams skipped"
        lines.append(basic_stats)

        # Pre-check stats (US-008)
        pre_check_total = self.pre_check_available + self.pre_check_unavailable
        if pre_check_total > 0:
            lines.append(
                f"  Pre-check: {self.pre_check_available} available, "
                f"{self.pre_check_unavailable} unavailable"
            )

        # Batch pre-check stats (US-006 Sprint 7)
        if self.pre_check_batched_total > 0:
            lines.append(
                f"  Batch pre-check: {self.pre_check_batched_checked} checked, "
                f"{self.pre_check_batched_skipped} skipped by pattern, "
                f"{self.pre_check_api_calls_saved} API calls saved"
            )

        # US-59-006: Unavailable summary
        if self.unavailable_count > 0:
            lines.append(
                f"  Unavailable: {self.unavailable_count} videos "
                f"({self.unavailable_rate:.1%} of fetched)"
            )

        if self.total_processed > 0:
            lines.append(
                f"  Success rate: {self.success_rate}%, "
                f"Cache hit rate: {self.cache_hit_rate}%"
            )
            lines.append(f"  Total segments: {self.total_segments}")

        # Caption source breakdown
        if self.human_caption_count > 0 or self.auto_generated_count > 0:
            source_line = (
                f"  Caption sources: {self.human_caption_count} human, "
                f"{self.auto_generated_count} auto, "
                f"{self.quality_distribution.get('unavailable', 0)} unavailable"
            )
            # US-62-005: Include auto-fallback count if any
            if self.auto_fallback_count > 0:
                source_line += f" ({self.auto_fallback_count} auto-fallback)"
            lines.append(source_line)

        # Language distribution (top 5)
        if self.language_distribution:
            sorted_langs = sorted(
                self.language_distribution.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            lang_str = ", ".join(f"{lang}: {count}" for lang, count in sorted_langs)
            lines.append(f"  Languages: {lang_str}")

        # Quality distribution
        if self.quality_distribution:
            quality_str = ", ".join(
                f"{q}: {c}" for q, c in sorted(self.quality_distribution.items())
            )
            lines.append(f"  Quality: {quality_str}")

        # Coverage distribution (US-004)
        if self.coverage_distribution:
            coverage_str = ", ".join(
                f"{level}: {count}" for level, count in sorted(self.coverage_distribution.items())
            )
            lines.append(f"  Coverage: {coverage_str}")

        # Low coverage warning (US-004)
        if self.low_coverage_videos:
            lines.append(f"  Low coverage: {len(self.low_coverage_videos)} videos below threshold")

        # Slowest fetches (US-002 Sprint 6)
        slowest = self.get_slowest_videos(5)
        if slowest:
            slowest_str = ", ".join(f"{vid}={t:.1f}s" for vid, t in slowest)
            lines.append(f"  Slowest fetches: {slowest_str}")

        # Language fallback summary (US-003 Sprint 6)
        if self.language_selection_trace:
            fallback_summary = self.get_language_fallback_summary()
            summary_parts = []
            if 'preferred' in fallback_summary:
                summary_parts.append(f"{fallback_summary['preferred']} preferred")
            fallback_keys = sorted(
                [k for k in fallback_summary if k.startswith('fallback_')],
                key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else 99
            )
            for key in fallback_keys:
                pos = key.replace('fallback_', 'fallback-')
                summary_parts.append(f"{fallback_summary[key]} {pos}")
            if 'english_fallback' in fallback_summary:
                summary_parts.append(f"{fallback_summary['english_fallback']} english-fallback")
            if 'any_available' in fallback_summary:
                summary_parts.append(f"{fallback_summary['any_available']} any-available")
            if 'none' in fallback_summary:
                summary_parts.append(f"{fallback_summary['none']} none")
            if 'other' in fallback_summary:
                summary_parts.append(f"{fallback_summary['other']} other")

            if summary_parts:
                lines.append(f"  Language fallback: {', '.join(summary_parts)}")

        # Format success statistics (US-004 Sprint 6)
        if self.format_success_counts:
            format_stats = self.get_format_statistics()
            format_parts = []
            sorted_formats = sorted(
                format_stats['format_counts'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for fmt, count in sorted_formats:
                rate = format_stats['format_rates'].get(fmt, 0.0)
                total = format_stats['total_with_format']
                format_parts.append(f"{fmt} {rate:.0f}% ({count}/{total})")

            if format_parts:
                format_line = f"  Format success: {', '.join(format_parts)}"
                if format_stats['fallback_count'] > 0:
                    format_line += f" ({format_stats['fallback_count']} fallback)"
                lines.append(format_line)

        # Channel availability summary (US-009 Sprint 7)
        channel_summary = self.get_channel_summary()
        if channel_summary:
            lines.append(f"  {channel_summary}")

        # Performance summary (US-59-009)
        if self.format_attempt_records or self.calls_saved_by_preflight or self.calls_saved_by_negative_cache:
            perf = self.get_performance_summary()
            perf_parts = [f"{perf['total_subprocess_calls']} subprocess calls"]
            if perf['total_subprocess_seconds'] > 0:
                perf_parts.append(f"{perf['total_subprocess_seconds']:.1f}s total")
                perf_parts.append(f"{perf['avg_seconds_per_call']:.2f}s avg")
            saved_parts = []
            if perf['calls_saved_by_preflight'] > 0:
                saved_parts.append(f"{perf['calls_saved_by_preflight']} by preflight")
            if perf['calls_saved_by_negative_cache'] > 0:
                saved_parts.append(f"{perf['calls_saved_by_negative_cache']} by neg-cache")
            if saved_parts:
                perf_parts.append(f"saved: {', '.join(saved_parts)}")
            lines.append(f"  Performance: {', '.join(perf_parts)}")

        return "\n".join(lines)

    def get_slowest_videos(self, n: int = 5) -> List[tuple]:
        """Get the n slowest video fetches by elapsed time.

        Thread-safe: Protected by lock for parallel fetching.

        Args:
            n: Number of slowest videos to return (default: 5).

        Returns:
            List of (video_id, elapsed_seconds) tuples sorted by time descending.
        """
        with self._lock:
            sorted_times = sorted(
                self.video_fetch_times.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return sorted_times[:n]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics to dictionary for checkpoint/JSON.

        Returns:
            Dict representation of all metrics.
        """
        # Import here to avoid circular import
        from .cache_models import ChannelCaptionPattern

        return {
            'fetch_attempts': self.fetch_attempts,
            'successes': self.successes,
            'failures': self.failures,
            'cache_hits': self.cache_hits,
            'skipped_live_streams': self.skipped_live_streams,
            'pre_check_available': self.pre_check_available,
            'pre_check_unavailable': self.pre_check_unavailable,
            'pre_check_batched_total': self.pre_check_batched_total,
            'pre_check_batched_skipped': self.pre_check_batched_skipped,
            'pre_check_batched_checked': self.pre_check_batched_checked,
            'pre_check_api_calls_saved': self.pre_check_api_calls_saved,
            'language_distribution': dict(self.language_distribution),
            'quality_distribution': dict(self.quality_distribution),
            'coverage_distribution': dict(self.coverage_distribution),
            'low_coverage_videos': list(self.low_coverage_videos),
            'video_fetch_times': dict(self.video_fetch_times),
            'language_selection_trace': list(self.language_selection_trace),
            'format_success_counts': dict(self.format_success_counts),
            'format_fallback_count': self.format_fallback_count,
            'video_format_used': dict(self.video_format_used),
            'cache_validation_passed': self.cache_validation_passed,
            'cache_validation_rejected': self.cache_validation_rejected,
            'cache_validation_refetched': self.cache_validation_refetched,
            'total_segments': self.total_segments,
            'auto_generated_count': self.auto_generated_count,
            'human_caption_count': self.human_caption_count,
            'error_category_counts': dict(self.error_category_counts),
            'channel_patterns': {
                cid: pattern.to_dict()
                for cid, pattern in self.channel_patterns.items()
            },
            'unavailable_count': self.unavailable_count,
            'unavailable_rate': self.unavailable_rate,
            'format_attempt_records': list(self.format_attempt_records),  # US-59-009
            'calls_saved_by_preflight': self.calls_saved_by_preflight,  # US-59-009
            'calls_saved_by_negative_cache': self.calls_saved_by_negative_cache,  # US-59-009
            'auto_fallback_count': self.auto_fallback_count,  # US-62-005
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CaptionMetrics':
        """Create CaptionMetrics from dictionary.

        Args:
            data: Dict from to_dict() or checkpoint.

        Returns:
            New CaptionMetrics instance.
        """
        if not data:
            return cls()

        # Import here to avoid circular import
        from .cache_models import ChannelCaptionPattern

        return cls(
            fetch_attempts=data.get('fetch_attempts', 0),
            successes=data.get('successes', 0),
            failures=data.get('failures', 0),
            cache_hits=data.get('cache_hits', 0),
            skipped_live_streams=data.get('skipped_live_streams', 0),
            pre_check_available=data.get('pre_check_available', 0),
            pre_check_unavailable=data.get('pre_check_unavailable', 0),
            pre_check_batched_total=data.get('pre_check_batched_total', 0),
            pre_check_batched_skipped=data.get('pre_check_batched_skipped', 0),
            pre_check_batched_checked=data.get('pre_check_batched_checked', 0),
            pre_check_api_calls_saved=data.get('pre_check_api_calls_saved', 0),
            language_distribution=data.get('language_distribution', {}),
            quality_distribution=data.get('quality_distribution', {}),
            coverage_distribution=data.get('coverage_distribution', {}),
            low_coverage_videos=data.get('low_coverage_videos', []),
            video_fetch_times=data.get('video_fetch_times', {}),
            language_selection_trace=data.get('language_selection_trace', []),
            format_success_counts=data.get('format_success_counts', {}),
            format_fallback_count=data.get('format_fallback_count', 0),
            video_format_used=data.get('video_format_used', {}),
            cache_validation_passed=data.get('cache_validation_passed', 0),
            cache_validation_rejected=data.get('cache_validation_rejected', 0),
            cache_validation_refetched=data.get('cache_validation_refetched', 0),
            total_segments=data.get('total_segments', 0),
            auto_generated_count=data.get('auto_generated_count', 0),
            human_caption_count=data.get('human_caption_count', 0),
            error_category_counts=data.get('error_category_counts', {}),
            channel_patterns={
                cid: ChannelCaptionPattern.from_dict(pattern_data)
                for cid, pattern_data in data.get('channel_patterns', {}).items()
            },
            unavailable_count=data.get('unavailable_count', 0),
            unavailable_rate=data.get('unavailable_rate', 0.0),
            format_attempt_records=data.get('format_attempt_records', []),  # US-59-009
            calls_saved_by_preflight=data.get('calls_saved_by_preflight', 0),  # US-59-009
            calls_saved_by_negative_cache=data.get('calls_saved_by_negative_cache', 0),  # US-59-009
            auto_fallback_count=data.get('auto_fallback_count', 0),  # US-62-005
        )

    def export_json(
        self,
        path: str,
        project_path: Optional[str] = None,
        config: Any = None,
        video_count: Optional[int] = None
    ) -> Dict[str, Any]:
        """Export all metrics to a timestamped JSON file (US-004 Sprint 7).

        Args:
            path: Output file path for JSON export.
            project_path: Optional project directory path for metadata.
            config: Optional Config object to include caption config snapshot.
            video_count: Optional count of videos processed for metadata.

        Returns:
            Dict with the exported data (same as written to file).

        Raises:
            OSError: If file cannot be written.
        """
        from datetime import datetime, timezone
        from pathlib import Path as PathLib
        import json

        export_timestamp = datetime.now(timezone.utc).isoformat()

        # Build config snapshot if config provided
        config_snapshot = None
        if config is not None:
            try:
                caption_config = getattr(config.download, 'caption_first', None)
                if caption_config:
                    config_snapshot = {
                        'enabled': getattr(caption_config, 'enabled', False),
                        'preferred_language': getattr(caption_config, 'preferred_language', 'en'),
                        'fallback_languages': getattr(caption_config, 'fallback_languages', []),
                        'preferred_formats': getattr(caption_config, 'preferred_formats', []),
                        'allow_auto_generated': getattr(caption_config, 'allow_auto_generated', True),
                        'min_coverage_threshold': getattr(caption_config, 'min_coverage_threshold', 0.5),
                        'max_fetch_timeout': getattr(caption_config, 'max_fetch_timeout', 30),
                        'adaptive_format_order': getattr(caption_config, 'adaptive_format_order', True),
                    }
            except (AttributeError, TypeError):
                config_snapshot = None

        # Get format statistics
        format_stats = self.get_format_statistics()

        # Get error category summary
        error_summary = self.get_error_category_summary()

        # Get language fallback summary
        lang_fallback_summary = self.get_language_fallback_summary()

        # Build structured export
        export_data = {
            "schema_version": "1.0",
            "export_timestamp": export_timestamp,
            "run_metadata": {
                "project_path": project_path,
                "video_count": video_count,
                "export_path": str(path),
            },
            "config_snapshot": config_snapshot,
            "summary": {
                "total_processed": self.total_processed,
                "success_rate_percent": self.success_rate,
                "cache_hit_rate_percent": self.cache_hit_rate,
                "fetch_attempts": self.fetch_attempts,
                "successes": self.successes,
                "failures": self.failures,
                "cache_hits": self.cache_hits,
                "total_segments": self.total_segments,
                "skipped_live_streams": self.skipped_live_streams,
            },
            "timing": {
                "video_fetch_times": dict(self.video_fetch_times),
                "slowest_videos": self.get_slowest_videos(10),
            },
            "formats": {
                "success_counts": format_stats['format_counts'],
                "success_rates": format_stats['format_rates'],
                "fallback_count": format_stats['fallback_count'],
                "fallback_rate_percent": format_stats['fallback_rate'],
                "video_format_used": dict(self.video_format_used),
            },
            "languages": {
                "distribution": dict(self.language_distribution),
                "selection_trace": list(self.language_selection_trace),
                "fallback_summary": lang_fallback_summary,
            },
            "errors": {
                "category_counts": error_summary['counts'],
                "category_rates": error_summary['category_rates'],
                "top_category": error_summary['top_category'],
            },
            "coverage": {
                "distribution": dict(self.coverage_distribution),
                "low_coverage_videos": list(self.low_coverage_videos),
            },
            "quality": {
                "distribution": dict(self.quality_distribution),
                "human_caption_count": self.human_caption_count,
                "auto_generated_count": self.auto_generated_count,
            },
            "cache_validation": {
                "passed": self.cache_validation_passed,
                "rejected": self.cache_validation_rejected,
                "refetched": self.cache_validation_refetched,
            },
            "pre_check": {
                "available": self.pre_check_available,
                "unavailable": self.pre_check_unavailable,
            },
            "raw_metrics": self.to_dict(),
        }

        # Write to file
        output_path = PathLib(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported caption metrics to {path}")

        return export_data

    def merge(self, other: 'CaptionMetrics') -> 'CaptionMetrics':
        """Merge another CaptionMetrics into this one.

        Thread-safe: Protected by lock for parallel fetching.

        Args:
            other: Another CaptionMetrics to merge.

        Returns:
            Self after merging.
        """
        with self._lock:
            self.fetch_attempts += other.fetch_attempts
            self.successes += other.successes
            self.failures += other.failures
            self.cache_hits += other.cache_hits
            self.skipped_live_streams += other.skipped_live_streams
            self.pre_check_available += other.pre_check_available
            self.pre_check_unavailable += other.pre_check_unavailable
            self.pre_check_batched_total += other.pre_check_batched_total
            self.pre_check_batched_skipped += other.pre_check_batched_skipped
            self.pre_check_batched_checked += other.pre_check_batched_checked
            self.pre_check_api_calls_saved += other.pre_check_api_calls_saved
            self.total_segments += other.total_segments
            self.auto_generated_count += other.auto_generated_count
            self.human_caption_count += other.human_caption_count
            self.auto_fallback_count += other.auto_fallback_count  # US-62-005

            # Merge language distribution
            for lang, count in other.language_distribution.items():
                self.language_distribution[lang] = self.language_distribution.get(lang, 0) + count

            # Merge quality distribution
            for quality, count in other.quality_distribution.items():
                self.quality_distribution[quality] = self.quality_distribution.get(quality, 0) + count

            # Merge coverage distribution
            for level, count in other.coverage_distribution.items():
                self.coverage_distribution[level] = self.coverage_distribution.get(level, 0) + count

            # Merge low coverage videos
            self.low_coverage_videos.extend(other.low_coverage_videos)

            # Merge video fetch times - keep slower time if duplicate
            for vid, elapsed in other.video_fetch_times.items():
                if vid not in self.video_fetch_times or elapsed > self.video_fetch_times[vid]:
                    self.video_fetch_times[vid] = elapsed

            # Merge language selection trace
            self.language_selection_trace.extend(other.language_selection_trace)

            # Merge format success counts
            for fmt, count in other.format_success_counts.items():
                self.format_success_counts[fmt] = self.format_success_counts.get(fmt, 0) + count
            self.format_fallback_count += other.format_fallback_count

            # Merge video format used - keep first occurrence
            for vid, fmt in other.video_format_used.items():
                if vid not in self.video_format_used:
                    self.video_format_used[vid] = fmt

            # Merge cache validation counts
            self.cache_validation_passed += other.cache_validation_passed
            self.cache_validation_rejected += other.cache_validation_rejected
            self.cache_validation_refetched += other.cache_validation_refetched

        return self

    def clear(self) -> None:
        """Reset all metrics.

        Thread-safe: Protected by lock for parallel fetching.
        """
        with self._lock:
            self.fetch_attempts = 0
            self.successes = 0
            self.failures = 0
            self.cache_hits = 0
            self.skipped_live_streams = 0
            self.pre_check_available = 0
            self.pre_check_unavailable = 0
            self.pre_check_batched_total = 0
            self.pre_check_batched_skipped = 0
            self.pre_check_batched_checked = 0
            self.pre_check_api_calls_saved = 0
            self.language_distribution = {}
            self.quality_distribution = {}
            self.coverage_distribution = {}
            self.low_coverage_videos = []
            self.video_fetch_times = {}
            self.language_selection_trace = []
            self.format_success_counts = {}
            self.format_fallback_count = 0
            self.video_format_used = {}
            self.cache_validation_passed = 0
            self.cache_validation_rejected = 0
            self.cache_validation_refetched = 0
            self.total_segments = 0
            self.auto_generated_count = 0
            self.human_caption_count = 0
            self.auto_fallback_count = 0  # US-62-005
