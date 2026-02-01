"""
Batch processing for caption fetching.

Extracts batch processing logic from CaptionFetcher to a dedicated class.
Provides progress callbacks, checkpoint support, and parallel execution.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Protocol, Union

if TYPE_CHECKING:
    from ..caption_fetcher import CaptionFetcher
    from .batch_checkpoint import CaptionBatchCheckpoint
    from .circuit_breaker import CaptionCircuitBreaker
    from .error_handling import ErrorPatternDetector
    from .metrics import CaptionMetrics
    from .models import CaptionResult, ErrorPatternResult
    from .retry_budget import CaptionRetryBudget
    from .worker_progress import WorkerProgressTracker
    from ..rate_limit.coordinator import GlobalRateLimitCoordinator

logger = logging.getLogger(__name__)


class ProgressCallback(Protocol):
    """Protocol for progress callbacks during batch processing."""

    def __call__(
        self,
        video_id: str,
        status: str,
        details: Dict[str, Any],
    ) -> None:
        """Called for each video progress update.

        Args:
            video_id: The video being processed (empty for batch-level events).
            status: One of 'fetching', 'success', 'failed', 'skipped', 'pattern_detected'.
            details: Additional context with fields like:
                - index: Current video number (1-based)
                - total: Total videos in batch
                - language: Caption language (on success)
                - quality: Caption quality (on success)
                - segment_count: Number of segments (on success)
                - is_auto_generated: Whether auto-generated (on success)
                - error: Error message (on failure)
                - reason: Failure reason (on failure/skip)
                - elapsed_seconds: Fetch duration
        """
        ...


@dataclass
class BatchProcessorConfig:
    """Configuration for BatchProcessor."""

    max_workers: int = 4
    checkpoint_save_interval: int = 10
    error_pattern_mode: str = 'warn'  # 'warn', 'abort', 'skip'
    error_pattern_threshold: float = 0.3
    error_pattern_sample_size: int = 10
    prioritize_by_channel: bool = True
    stuck_worker_threshold: float = 60.0
    slow_threshold_ratio: float = 0.8  # 80% of timeout = slow


@dataclass
class BatchResult:
    """Result of batch processing operation."""

    results: Dict[str, Any] = field(default_factory=dict)
    success_count: int = 0
    error_count: int = 0
    skipped_count: int = 0
    aborted: bool = False
    abort_reason: Optional[str] = None
    pattern_detected: Optional['ErrorPatternResult'] = None


class BatchProcessor:
    """Processes caption fetches in parallel batches.

    Extracted from CaptionFetcher.fetch_captions_batch() to provide a dedicated
    class for batch processing with progress callbacks.

    Features:
    - Parallel execution using ThreadPoolExecutor
    - Progress callbacks via on_video_complete
    - Checkpoint support for resumable batches
    - Error pattern detection
    - Circuit breaker integration
    - Retry budget tracking
    - Rate limit coordination

    Example:
        fetcher = CaptionFetcher(config)
        processor = BatchProcessor(fetcher, config=BatchProcessorConfig(max_workers=8))

        def on_complete(video_id: str, result: Any) -> None:
            print(f"Completed: {video_id}")

        batch_result = processor.process(
            video_ids=['abc123', 'def456'],
            on_video_complete=on_complete,
        )
        print(f"Success: {batch_result.success_count}, Errors: {batch_result.error_count}")
    """

    def __init__(
        self,
        fetcher: 'CaptionFetcher',
        config: Optional[BatchProcessorConfig] = None,
    ):
        """Initialize BatchProcessor.

        Args:
            fetcher: CaptionFetcher instance for individual caption fetches.
            config: Optional configuration. Defaults to BatchProcessorConfig().
        """
        self.fetcher = fetcher
        self.config = config or BatchProcessorConfig()

        # Internal state
        self._results: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def process(
        self,
        video_ids: List[str],
        preferred_language: Optional[str] = None,
        metrics: Optional['CaptionMetrics'] = None,
        progress_callback: Optional[ProgressCallback] = None,
        on_video_complete: Optional[Callable[[str, Any], None]] = None,
        skip_video_ids: Optional[set] = None,
        batch_checkpoint: Optional['CaptionBatchCheckpoint'] = None,
        circuit_breaker: Optional['CaptionCircuitBreaker'] = None,
        retry_budget: Optional['CaptionRetryBudget'] = None,
        rate_limit_coordinator: Optional['GlobalRateLimitCoordinator'] = None,
    ) -> BatchResult:
        """Process a batch of videos for caption fetching.

        Args:
            video_ids: List of YouTube video IDs to fetch captions for.
            preferred_language: Preferred caption language (ISO 639-1 code).
            metrics: Optional CaptionMetrics instance for thread-safe tracking.
            progress_callback: Optional callback for detailed progress updates.
            on_video_complete: Simple callback called when each video completes.
                Signature: (video_id: str, result: CaptionResult | dict) -> None
            skip_video_ids: Set of video IDs to skip (already cached).
            batch_checkpoint: Optional checkpoint for partial result recovery.
            circuit_breaker: Optional circuit breaker for pause on failures.
            retry_budget: Optional retry budget for resource tracking.
            rate_limit_coordinator: Optional global rate limiter.

        Returns:
            BatchResult with all results and summary stats.

        Raises:
            ErrorPatternAbortError: When abort mode is set and pattern detected.
        """
        from .exceptions import ErrorPatternAbortError
        from .error_handling import ErrorPatternDetector
        from .worker_progress import WorkerProgressTracker
        from .models import CaptionResult

        if not video_ids:
            return BatchResult()

        # Filter out skipped video IDs
        skip_set = skip_video_ids or set()
        videos_to_fetch = [vid for vid in video_ids if vid not in skip_set]

        # Prioritize by channel success rate if enabled
        if self.config.prioritize_by_channel and metrics and metrics.channel_patterns:
            videos_to_fetch = self.fetcher._sort_videos_by_channel_success(
                videos_to_fetch, metrics.channel_patterns
            )
            logger.info(f"Prioritized fetch order by channel success rate")

        logger.info(
            f"Batch caption fetch: {len(videos_to_fetch)} videos with "
            f"{self.config.max_workers} workers ({len(skip_set)} skipped)"
        )

        # Initialize tracking
        self._results = {}
        total_videos = len(videos_to_fetch)
        checkpoint_success_count = 0
        checkpoint_lock = threading.Lock()

        # Setup error pattern detection
        pattern_detector: Optional[ErrorPatternDetector] = None
        if self.config.error_pattern_mode != 'skip':
            pattern_detector = ErrorPatternDetector(
                threshold=self.config.error_pattern_threshold,
                sample_size=self.config.error_pattern_sample_size
            )

        # Setup worker progress tracker
        worker_tracker = WorkerProgressTracker(
            total_videos=total_videos,
            stuck_threshold_seconds=self.config.stuck_worker_threshold,
            max_fetch_times=10,
        )
        worker_tracker.initialize_workers(self.config.max_workers)

        # Thread to worker mapping
        thread_to_worker: Dict[int, int] = {}
        thread_lock = threading.Lock()
        next_worker_id = [0]

        def get_worker_id() -> int:
            thread_id = threading.get_ident()
            with thread_lock:
                if thread_id not in thread_to_worker:
                    thread_to_worker[thread_id] = next_worker_id[0]
                    next_worker_id[0] += 1
                return thread_to_worker[thread_id]

        # Calculate slow threshold
        timeout = self.fetcher._timeout if self.fetcher._timeout else 30.0
        slow_threshold = timeout * self.config.slow_threshold_ratio

        def fetch_single(video_id: str, index: int) -> tuple:
            """Fetch captions for a single video."""
            start_time = time.perf_counter()

            # Check circuit breaker
            if circuit_breaker and circuit_breaker.is_enabled:
                circuit_breaker.check_and_wait()

            # Acquire rate limit slot
            slot_acquired = False
            if rate_limit_coordinator and rate_limit_coordinator.is_enabled():
                slot_acquired = rate_limit_coordinator.acquire_slot(
                    'caption', timeout=30.0, block=True
                )
                if not slot_acquired:
                    logger.warning(f"Rate limit slot timeout for {video_id}")
                    skip_result = {
                        'video_id': video_id,
                        'skipped': True,
                        'reason': 'rate_limit_timeout',
                        'caption_quality': 'low',
                    }
                    _notify_progress(video_id, 'skipped', {
                        'index': index + 1,
                        'total': total_videos,
                        'reason': 'rate_limit_timeout',
                    })
                    return video_id, skip_result

            # Check retry budget
            if retry_budget and retry_budget.budget_exhausted():
                retry_budget.record_skipped(video_id)
                skip_result = {
                    'video_id': video_id,
                    'skipped': True,
                    'reason': 'budget_exhausted',
                    'caption_quality': 'low',
                }
                _notify_progress(video_id, 'skipped', {
                    'index': index + 1,
                    'total': total_videos,
                    'reason': 'budget_exhausted',
                })
                return video_id, skip_result

            # Record attempt
            if retry_budget:
                retry_budget.record_attempt(video_id)

            # Track worker start
            worker_id = get_worker_id()
            worker_tracker.worker_start(worker_id, video_id)

            # Notify progress: fetching
            _notify_progress(video_id, 'fetching', {
                'index': index + 1,
                'total': total_videos,
                'worker_id': worker_id,
            })

            # Record fetch attempt in metrics
            if metrics:
                metrics.record_fetch_attempt(video_id)

            try:
                result = self.fetcher.fetch_captions_auto_language_with_retry(
                    video_id,
                    preferred_language=preferred_language
                )

                elapsed = time.perf_counter() - start_time

                # Warn if slow
                if elapsed > slow_threshold:
                    logger.warning(
                        f"Slow caption fetch for {video_id}: {elapsed:.1f}s "
                        f"(>{slow_threshold:.1f}s threshold)"
                    )

                # Record success in metrics
                if metrics:
                    metrics.record_fetch_success(
                        video_id=video_id,
                        language=result.language,
                        quality=result.caption_quality,
                        segment_count=len(result.segments),
                        is_auto_generated=result.is_auto_generated,
                        elapsed_seconds=elapsed,
                        format_source=result.format_source,
                        preferred_format=(
                            self.fetcher._preferred_formats[0]
                            if self.fetcher._preferred_formats else None
                        ),
                    )

                # Mark worker complete
                worker_tracker.worker_complete(worker_id)

                # Record success in circuit breaker
                if circuit_breaker and circuit_breaker.is_enabled:
                    circuit_breaker.record_success()

                # Record success in retry budget
                if retry_budget:
                    retry_budget.record_success(video_id)

                # Notify progress: success
                _notify_progress(video_id, 'success', {
                    'index': index + 1,
                    'total': total_videos,
                    'language': result.language,
                    'quality': result.caption_quality,
                    'segment_count': len(result.segments),
                    'is_auto_generated': result.is_auto_generated,
                    'elapsed_seconds': elapsed,
                    'format_source': result.format_source,
                    'worker_id': worker_id,
                })

                return video_id, result

            except Exception as e:
                elapsed = time.perf_counter() - start_time
                worker_tracker.worker_complete(worker_id)

                # Determine error type
                from .exceptions import CaptionUnavailableError, CaptionFetchError

                if isinstance(e, CaptionUnavailableError):
                    error_result = {
                        'video_id': video_id,
                        'unavailable': True,
                        'reason': str(e.reason),
                        'caption_quality': 'low',
                        'elapsed_seconds': elapsed,
                    }
                    reason = 'unavailable'
                else:
                    # Record failure in circuit breaker (fetch errors trip it)
                    if circuit_breaker and circuit_breaker.is_enabled:
                        if isinstance(e, CaptionFetchError):
                            circuit_breaker.record_failure()

                    # Record failure in retry budget
                    if retry_budget:
                        retry_budget.record_failure(video_id)

                    error_result = {
                        'video_id': video_id,
                        'error': True,
                        'reason': str(getattr(e, 'reason', str(e))),
                        'caption_quality': 'low',
                        'elapsed_seconds': elapsed,
                    }
                    reason = 'error'

                    if not isinstance(e, CaptionFetchError):
                        logger.exception(f"Unexpected error fetching {video_id}")

                # Record failure in metrics
                if metrics:
                    metrics.record_fetch_failure(
                        video_id=video_id,
                        reason=reason,
                        elapsed_seconds=elapsed
                    )

                # Notify progress: failed
                _notify_progress(video_id, 'failed', {
                    'index': index + 1,
                    'total': total_videos,
                    'reason': reason,
                    'error': str(getattr(e, 'reason', str(e))),
                    'elapsed_seconds': elapsed,
                    'worker_id': worker_id,
                })

                return video_id, error_result

            finally:
                # Release rate limit slot
                if slot_acquired and rate_limit_coordinator:
                    rate_limit_coordinator.release_slot('caption')

        def _is_caption_result(result: Any) -> bool:
            """Check if result is a successful CaptionResult (duck typing).

            Uses duck typing to detect CaptionResult objects. This allows
            mock objects in tests to be treated as successes.
            """
            # Check for CaptionResult-like object (has segments, language, video_id)
            return (
                hasattr(result, 'segments')
                and hasattr(result, 'language')
                and hasattr(result, 'video_id')
            )

        def _notify_progress(video_id: str, status: str, details: Dict) -> None:
            """Safely call progress callback."""
            if progress_callback:
                try:
                    progress_callback(video_id, status, details)
                except Exception as e:
                    logger.debug(f"Progress callback error: {e}")

        def _notify_complete(video_id: str, result: Any) -> None:
            """Call on_video_complete callback."""
            if on_video_complete:
                try:
                    on_video_complete(video_id, result)
                except Exception as e:
                    logger.debug(f"on_video_complete callback error: {e}")

        # Track pattern handling
        pattern_handled = False
        batch_result = BatchResult()

        # Execute fetches in parallel
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(fetch_single, video_id, idx): video_id
                for idx, video_id in enumerate(videos_to_fetch)
            }

            for future in as_completed(futures):
                video_id = futures[future]
                try:
                    vid, result = future.result()
                    self._results[vid] = result

                    # Update counts (use duck typing to check for success)
                    is_success = _is_caption_result(result)
                    if is_success:
                        batch_result.success_count += 1
                    elif isinstance(result, dict) and result.get('skipped'):
                        batch_result.skipped_count += 1
                    else:
                        batch_result.error_count += 1

                    # Notify on_video_complete callback
                    _notify_complete(vid, result)

                    # Update checkpoint
                    if batch_checkpoint is not None:
                        with checkpoint_lock:
                            batch_checkpoint.update(vid, result)
                            if is_success:
                                checkpoint_success_count += 1
                                if checkpoint_success_count % self.config.checkpoint_save_interval == 0:
                                    logger.debug(
                                        f"Checkpoint: {checkpoint_success_count} successes "
                                        f"({len(batch_checkpoint.results)}/{total_videos} processed)"
                                    )

                    # Error pattern detection
                    if pattern_detector and not pattern_handled:
                        if is_success:
                            pattern_detector.record_success(vid)
                        else:
                            error_reason = result.get('reason', 'unknown error')
                            pattern_detector.record_error(vid, error_reason)

                        if pattern_detector.should_check_pattern():
                            pattern_result = pattern_detector.check_pattern()

                            if pattern_result.detected:
                                logger.warning(
                                    f"Error pattern detected: {pattern_result.error_signature} "
                                    f"({len(pattern_result.affected_video_ids)}/{pattern_result.sample_size} videos)"
                                )

                                if metrics:
                                    metrics.record_error_pattern_detected(
                                        pattern_result.error_signature,
                                        len(pattern_result.affected_video_ids),
                                        pattern_result.sample_size
                                    )

                                _notify_progress('', 'pattern_detected', {
                                    'error_signature': pattern_result.error_signature,
                                    'affected_count': len(pattern_result.affected_video_ids),
                                    'sample_size': pattern_result.sample_size,
                                    'ratio': pattern_result.ratio,
                                    'likely_cause': pattern_result.likely_cause,
                                })

                                if self.config.error_pattern_mode == 'abort':
                                    logger.error(f"Aborting batch: {pattern_result}")

                                    # Cancel remaining futures
                                    for f in futures:
                                        f.cancel()

                                    # Save checkpoint before abort
                                    if batch_checkpoint is not None:
                                        processed = set(self._results.keys())
                                        remaining = [v for v in videos_to_fetch if v not in processed]
                                        batch_checkpoint.mark_aborted(
                                            reason=str(pattern_result),
                                            pattern_result=pattern_result,
                                            remaining_ids=remaining
                                        )

                                    raise ErrorPatternAbortError(
                                        pattern_result=pattern_result,
                                        partial_results=dict(self._results)
                                    )

                            pattern_handled = True

                except ErrorPatternAbortError:
                    raise
                except Exception as e:
                    logger.error(f"Batch fetch future error for {video_id}: {e}")
                    self._results[video_id] = {
                        'video_id': video_id,
                        'error': True,
                        'reason': str(e),
                        'caption_quality': 'low',
                    }
                    batch_result.error_count += 1

                    if pattern_detector and not pattern_handled:
                        pattern_detector.record_error(video_id, str(e))

        logger.info(
            f"Batch caption fetch complete: {len(self._results)} processed, "
            f"{batch_result.success_count} succeeded"
        )

        batch_result.results = dict(self._results)
        return batch_result

    def get_results(self) -> Dict[str, Any]:
        """Get current results (for inspection during processing)."""
        with self._lock:
            return dict(self._results)
