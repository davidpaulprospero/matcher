"""
Parallel video transcription processor.

Orchestrates batch transcription with two-phase processing:
1. Parallel audio extraction (CPU-bound, I/O)
2. Sequential GPU transcription (shared WhisperModel)

Supports FFmpeg pipelining (US-124-012) to overlap audio extraction
with transcription for improved throughput.

Also provides single-video and voiceover transcription wrappers.
"""

import asyncio
import json
import logging
import shutil
import time
from collections import deque
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from .whisper_client import WhisperClient, _get_gpu_memory_mb, get_available_gpu_memory, get_gpu_utilization
from .whisper_client import run_transcription_health_checks
from .cache import TranscriptCache
from .delta_index import DeltaAwareIndex
from .utils import extract_audio, write_srt, get_audio_duration
from .exceptions import is_transient_error
from .metrics import TranscriptionMetrics
from .retry_budget import TranscriptionRetryBudget, TranscriptionBackoffManager, BackoffStrategy
from src.state import TranscriptSegment
from src.logging_templates import log_error_with_context

logger = logging.getLogger(__name__)


# US-137-006: Helper functions for pipeline optimization


def auto_tune_max_workers(config: Any = None) -> int:
    """
    Auto-tune max_workers based on CPU cores and GPU availability (US-137-006).

    Args:
        config: Configuration object with transcription settings

    Returns:
        Recommended number of workers for audio extraction
    """
    import os

    cpu_count = os.cpu_count() or 4

    # Check if GPU is available
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        pass

    # Base worker count from config
    default_workers = min(4, cpu_count)

    if config is None:
        return default_workers

    # Check if auto-tuning is enabled
    auto_tune = getattr(config.transcription, 'auto_tune_workers', True) if hasattr(config, 'transcription') else True

    if not auto_tune:
        # Use explicit setting from config
        return getattr(config.transcription, 'audio_extraction_workers', default_workers)

    # Calculate based on CPU cores and GPU
    # More workers for CPU-only (GPU not competing), fewer for GPU (GPU is bottleneck)
    worker_multiplier = getattr(config.transcription, 'worker_multiplier', 0.5) if hasattr(config, 'transcription') else 0.5

    if gpu_available:
        # With GPU, reduce worker count to avoid I/O bottleneck competing with GPU
        recommended = max(1, int(cpu_count * worker_multiplier * 0.5))
    else:
        # CPU-only, can use more workers for I/O parallelism
        recommended = max(1, int(cpu_count * worker_multiplier))

    logger.debug(f"Auto-tuned max_workers: {recommended} (CPU cores: {cpu_count}, GPU available: {gpu_available})")
    return recommended


def adjust_pipeline_depth(
    current_depth: int,
    gpu_utilization: float,
    config: Any = None
) -> int:
    """
    Dynamically adjust pipeline depth based on GPU utilization (US-137-006).

    Args:
        current_depth: Current pipeline depth
        gpu_utilization: Current GPU utilization percentage (0-100), or -1 if unavailable
        config: Configuration object with transcription settings

    Returns:
        Adjusted pipeline depth
    """
    if config is None:
        return current_depth

    # Check if dynamic adjustment is enabled
    dynamic_enabled = getattr(config.transcription, 'dynamic_pipeline_depth', True) if hasattr(config, 'transcription') else True

    if not dynamic_enabled:
        return current_depth

    # Get thresholds from config
    high_threshold = getattr(config.transcription, 'gpu_utilization_threshold_high', 85.0) if hasattr(config, 'transcription') else 85.0
    low_threshold = getattr(config.transcription, 'gpu_utilization_threshold_low', 50.0) if hasattr(config, 'transcription') else 50.0
    min_depth = getattr(config.transcription, 'min_pipeline_depth', 1) if hasattr(config, 'transcription') else 1
    max_depth = getattr(config.transcription, 'max_pipeline_depth', 6) if hasattr(config, 'transcription') else 6

    # If GPU utilization is unavailable, return current depth
    if gpu_utilization < 0:
        logger.debug(f"GPU utilization unavailable, keeping pipeline_depth={current_depth}")
        return current_depth

    new_depth = current_depth

    if gpu_utilization > high_threshold:
        # High GPU utilization - reduce pipeline depth to avoid queue buildup
        new_depth = max(min_depth, current_depth - 1)
    elif gpu_utilization < low_threshold:
        # Low GPU utilization - can increase pipeline depth for better overlap
        new_depth = min(max_depth, current_depth + 1)

    if new_depth != current_depth:
        logger.info(f"Adjusted pipeline_depth: {current_depth} -> {new_depth} (GPU util: {gpu_utilization:.1f}%)")

    return new_depth


def _clear_cuda_cache() -> None:
    """
    Clear CUDA memory cache to help recover from GPU OOM errors.

    Called between retry attempts for transient GPU errors.
    No-op if torch is not installed or CUDA is not available.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("Cleared CUDA cache between retry attempts")
    except ImportError:
        # torch not installed - using CPU-only mode
        pass
    except Exception as e:
        # Don't let cache clearing failure block retry
        logger.debug(f"Could not clear CUDA cache: {e}")


# Type alias for return value with metrics
TranscriptionResult = Tuple[Dict[str, List[TranscriptSegment]], TranscriptionMetrics]


# US-124-012: Async audio extraction for pipelining
async def _extract_audio_async(video_path: str, output_dir: str, timeout: int) -> Tuple[str, Optional[str]]:
    """
    Async wrapper for audio extraction.

    Runs blocking FFmpeg extraction in a thread pool to avoid blocking the event loop.
    This allows overlapping audio extraction with transcription.

    Args:
        video_path: Path to video file
        output_dir: Directory for output audio file
        timeout: Maximum time in seconds for extraction

    Returns:
        Tuple of (video_path, audio_path or None if extraction failed)
    """
    loop = asyncio.get_event_loop()
    audio_path = await loop.run_in_executor(
        None,  # Use default executor (ThreadPool)
        extract_audio,
        video_path,
        output_dir,
        timeout
    )
    return video_path, audio_path


def transcribe_videos_parallel(
    video_paths: List[str],
    cache: Any,
    config: Any = None,
    max_workers: int = None,
    force_reprocess: bool = False,
    show_progress: bool = True,
    skip_if_cached: bool = True,
    return_metrics: bool = False,
    checkpoint: Any = None,
    transcription_progress_callback: Any = None,
    progress_interval: int = 5
) -> Union[Dict[str, List[TranscriptSegment]], TranscriptionResult]:
    """
    Transcribe multiple videos with parallel audio extraction but sequential GPU.

    TWO-PHASE PROCESSING:
    Phase 1: Parallel audio extraction (CPU-bound, safe to parallelize)
    Phase 2: Sequential GPU transcription (must be serialized)

    Args:
        video_paths: List of video file paths
        cache: CacheManager or similar with cache_dir attribute
        config: Configuration object with transcription settings
        max_workers: Number of parallel workers for audio extraction.
                    If None, uses config.transcription.audio_extraction_workers
                    (US-60-010), which defaults to min(4, cpu_count()).
        force_reprocess: If True, ignore cache and reprocess all (deprecated, use skip_if_cached=False)
        show_progress: Whether to show progress
        skip_if_cached: If True (default), skip videos already in cache.
                       If False, reprocess all videos ignoring cache.
        return_metrics: If True, return (results, metrics) tuple instead of just results.
                       Added in US-60-009 for batch transcription progress tracking.
        checkpoint: Optional CheckpointManager instance. If provided, persists
                   TranscriptionMetrics summary to checkpoint after batch completion.
                   Added in US-79-012 for cross-run comparison.
        transcription_progress_callback: Optional callback function for real-time progress updates.
                   Callback receives: (progress_pct, current_index, total_videos, video_name).
                   Added in US-110-002 for pipeline integration.
        progress_interval: Number of videos between callback invocations (default: 5).
                   Also fires at 10% progress intervals. Added in US-110-002.

    Returns:
        If return_metrics=False: Dict mapping video path to list of TranscriptSegments
        If return_metrics=True: Tuple of (results dict, TranscriptionMetrics)
    """
    # Get cache directory from cache object or use as string
    if hasattr(cache, 'cache_dir'):
        cache_dir = cache.cache_dir
    else:
        cache_dir = str(cache)

    # Get model settings from config
    if config:
        model_name = getattr(config.transcription, 'model', 'base')
        model_version = getattr(config.transcription, 'model_version', None)  # US-124-010
        compute_type = getattr(config.transcription, 'compute_type', 'auto')
        language = getattr(config.transcription, 'language', None)
        # VAD is ALWAYS disabled for video transcription - config setting is for voiceover only
        # YouTube video audio quality varies, VAD is too aggressive and removes speech
        vad_filter = False
        min_silence_duration_ms = getattr(config.transcription, 'min_silence_duration_ms', 200)
        speech_pad_ms = getattr(config.transcription, 'speech_pad_ms', 10)
        # Audio extraction workers (US-60-010)
        if max_workers is None:
            # US-137-006: Auto-tune workers based on CPU cores and GPU availability
            max_workers = auto_tune_max_workers(config)
    else:
        model_name = "base"
        model_version = None  # US-124-010
        compute_type = "auto"
        language = None
        vad_filter = False  # Default False when no config
        min_silence_duration_ms = 200
        speech_pad_ms = 10

    # Fallback if max_workers still None (no config provided)
    if max_workers is None:
        import os
        cpu_count = os.cpu_count() or 4
        max_workers = min(4, cpu_count)

    # Get auto_cleanup setting (US-60-011)
    auto_cleanup_after_batch = True  # Default enabled
    if config:
        auto_cleanup_after_batch = getattr(config.transcription, 'auto_cleanup_after_batch', True)

    # Get GPU transcription timeout (US-79-002)
    gpu_transcription_timeout = 300  # Default 5 minutes
    if config:
        gpu_transcription_timeout = getattr(config.transcription, 'gpu_transcription_timeout', 300)

    # Get audio extraction timeout (US-79-003)
    audio_extraction_timeout = 60  # Default 60 seconds
    if config:
        audio_extraction_timeout = getattr(config.transcription, 'audio_extraction_timeout', 60)

    # Get FFmpeg pipelining setting (US-124-012, US-137-006)
    # Pipeline depth: number of videos to extract audio for ahead of transcription
    # When > 0, audio extraction runs concurrently with transcription
    # Set to 0 to disable pipelining (traditional two-phase approach)
    # Default is now 3 (US-137-006)
    pipeline_depth = 3
    if config:
        pipeline_depth = getattr(config.transcription, 'pipeline_depth', 3)
        # Handle Mock objects in tests - ensure we get an actual int
        if not isinstance(pipeline_depth, int):
            pipeline_depth = 3

    # US-137-006: Get initial GPU utilization and adjust pipeline depth if dynamic enabled
    initial_gpu_util = get_gpu_utilization()
    if config and pipeline_depth > 0:
        pipeline_depth = adjust_pipeline_depth(pipeline_depth, initial_gpu_util, config)

    # Get max retries for transient errors (US-79-004)
    max_retries = 2  # Default 2 retries
    if config:
        max_retries = getattr(config.transcription, 'max_retries', 2)

    # Get Whisper model threading settings (US-79-007)
    whisper_num_workers = 1  # Default 1 for GPU serialization
    whisper_cpu_threads = 4  # Default 4 threads
    if config:
        whisper_num_workers = getattr(config.transcription, 'whisper_num_workers', 1)
        whisper_cpu_threads = getattr(config.transcription, 'whisper_cpu_threads', 4)

    # Get GPU-to-CPU fallback setting (US-110-004)
    auto_fallback_to_cpu = True  # Default enabled
    if config:
        auto_fallback_to_cpu = getattr(config.transcription, 'auto_fallback_to_cpu', True)

    # Get cache compression setting (US-110-008)
    compress_cache = True  # Default enabled
    if config:
        compress_cache = getattr(config.transcription, 'compress_cache', True)

    # Get segment quality filtering setting (US-110-009)
    min_segment_words = 3  # Default minimum words per segment
    if config:
        min_segment_words = getattr(config.transcription, 'min_segment_words', 3)

    # Get auto model selection setting (US-110-010)
    auto_model_selection = True  # Default enabled
    if config:
        auto_model_selection = getattr(config.transcription, 'auto_model_selection', True)

    # Get progress logging interval (US-79-008)
    progress_log_interval = 10  # Default every 10 items
    if config:
        progress_log_interval = getattr(config.transcription, 'progress_log_interval', 10)

    # Get callback progress interval (US-110-002)
    # Default: every 5 videos or 10% - whichever comes first
    callback_interval = progress_interval
    if config:
        callback_interval = getattr(config.transcription, 'progress_callback_interval', callback_interval)

    # Initialize batch retry budget (US-79-010)
    retry_budget_max_attempts = 50
    retry_budget_max_backoff_seconds = 180.0
    if config:
        retry_budget_max_attempts = getattr(config.transcription, 'retry_budget_max_attempts', 50)
        retry_budget_max_backoff_seconds = getattr(config.transcription, 'retry_budget_max_backoff_seconds', 180.0)

    # Get batch processing settings (US-110-005)
    # 0 means no batching (process all at once)
    video_batch_size = 0  # Default: process all videos in one batch
    batch_wait_seconds = 0  # Default: no wait between batches
    if config:
        video_batch_size = getattr(config.transcription, 'batch_size', 0)
        batch_wait_seconds = getattr(config.transcription, 'batch_wait_seconds', 0)

    retry_budget = TranscriptionRetryBudget(
        max_attempts=retry_budget_max_attempts,
        max_backoff_time=retry_budget_max_backoff_seconds,
    )

    # Initialize backoff manager with jitter correlation (US-137-011)
    backoff_strategy_str = "jitter"
    backoff_jitter_factor = 0.3
    backoff_correlation_factor = 0.5
    backoff_max_jitter_cap = 10.0
    if config:
        backoff_strategy_str = getattr(config.transcription, 'backoff_strategy', 'jitter')
        backoff_jitter_factor = getattr(config.transcription, 'backoff_jitter_factor', 0.3)
        backoff_correlation_factor = getattr(config.transcription, 'backoff_correlation_factor', 0.5)
        backoff_max_jitter_cap = getattr(config.transcription, 'backoff_max_jitter_cap', 10.0)

    # Convert string to BackoffStrategy enum
    strategy_map = {
        'standard': BackoffStrategy.STANDARD,
        'jitter': BackoffStrategy.JITTER,
        'correlated': BackoffStrategy.CORRELATED,
        'adaptive': BackoffStrategy.ADAPTIVE,
    }
    backoff_strategy_enum = strategy_map.get(backoff_strategy_str, BackoffStrategy.JITTER)

    backoff_manager = TranscriptionBackoffManager(
        base_delay=1.0,
        jitter_factor=backoff_jitter_factor,
        correlation_factor=backoff_correlation_factor,
        strategy=backoff_strategy_enum,
        max_jitter_cap=backoff_max_jitter_cap,
    )

    # Initialize WhisperClient and TranscriptCache
    whisper_client = WhisperClient(
        model_name=model_name, model_version=model_version, compute_type=compute_type,
        gpu_transcription_timeout=gpu_transcription_timeout,
        num_workers=whisper_num_workers,
        cpu_threads=whisper_cpu_threads,
        auto_fallback_to_cpu=auto_fallback_to_cpu,  # US-110-004
        auto_model_selection=auto_model_selection  # US-110-010
    )
    transcript_cache = TranscriptCache(
        cache_dir,
        compress_cache=compress_cache,  # US-110-008
        min_segment_words=min_segment_words  # US-110-009
    )
    results = {}

    # Delta index staleness check (US-79-011)
    delta_index = DeltaAwareIndex(cache_dir)
    cache_entry_count = len(transcript_cache._source_map)
    is_stale, indexed_count, actual_count = delta_index.check_staleness(cache_entry_count)
    if is_stale:
        logger.warning(
            f"Delta index is stale: index has {indexed_count} entries but cache has "
            f"{actual_count} entries. Rebuilding index from cache."
        )
        delta_index.rebuild_from_cache(list(transcript_cache._source_map.keys()))

    try:
        # Initialize metrics (US-60-009)
        metrics = TranscriptionMetrics(total_videos=len(video_paths))

        # Log batch processing start with item counts (US-169-011)
        logger.info(
            f"[TRANSCRIBE] Batch processing start: total_videos={len(video_paths)}, "
            f"skip_if_cached={skip_if_cached}, force_reprocess={force_reprocess}"
        )

        # Determine whether to skip cached videos
        # skip_if_cached takes precedence; force_reprocess is deprecated but still honored
        should_skip_cached = skip_if_cached and not force_reprocess

        # Separate cached vs uncached
        cached_videos = []
        uncached_videos = []

        for video_path in video_paths:
            if not should_skip_cached:
                # Reprocess all videos (skip_if_cached=False or force_reprocess=True)
                uncached_videos.append(video_path)
            else:
                # Check cache when skip_if_cached=True
                cached = transcript_cache.get(video_path)
                if cached:
                    cached_videos.append((video_path, cached))
                else:
                    uncached_videos.append(video_path)

        # Load cached results and record cache hits
        for video_path, segments in cached_videos:
            results[video_path] = [
                TranscriptSegment(
                    index=i,
                    start_time=seg['start'],
                    end_time=seg['end'],
                    text=seg['text'],
                    source_file=video_path
                )
                for i, seg in enumerate(segments)
            ]
            metrics.record_cache_hit(video_path)

        # Log skipped videos count when skip_if_cached=True
        if should_skip_cached and cached_videos:
            logger.info(f"Skipped {len(cached_videos)} cached videos (skip_if_cached=True)")

        if show_progress:
            logger.info(f"Video index: {len(cached_videos)} cached, {len(uncached_videos)} new")

        # US-110-005: Log batch configuration when processing large numbers
        if show_progress and video_batch_size > 0 and len(uncached_videos) > video_batch_size:
            logger.info(
                f"Batch processing enabled: {video_batch_size} videos per batch, "
                f"{batch_wait_seconds}s pause between batches"
            )

        if not uncached_videos:
            # Log metrics summary even when all cached (US-60-009)
            summary = metrics.get_summary_dict()
            logger.info(f"Transcription metrics: {summary}")

            # US-79-012: Persist metrics even when all cached
            if checkpoint is not None:
                try:
                    checkpoint_summary = {
                        'total_videos': summary['total_videos'],
                        'cached_videos': summary['cached_hits'],
                        'transcribed_videos': summary['transcribed_count'],
                        'failed_videos': summary['failed_count'],
                        'total_duration_seconds': summary['total_duration_s'],
                        'average_speed_ratio': summary['avg_speed_ratio'],
                        'cache_hit_rate': summary['cache_hit_rate'],
                        'success_rate': summary['success_rate'],
                        'phase1_time_s': 0.0,
                        'phase2_time_s': 0.0,
                    }
                    checkpoint.save_transcription_metrics(checkpoint_summary)
                except Exception as e:
                    logger.warning(f"Could not persist transcription metrics to checkpoint: {e}")

            if return_metrics:
                return results, metrics
            return results

        # Create temp directory for audio files
        temp_dir = Path(cache_dir) / "temp_audio"
        temp_dir.mkdir(parents=True, exist_ok=True)

        total_videos = len(uncached_videos)

        # US-137-007: Run transcription health checks before batch start
        if total_videos > 0:
            if show_progress:
                logger.info("Running transcription health checks...")
            health_result = run_transcription_health_checks(
                model_name=model_name,
                compute_type=compute_type,
                skip_model_check=False
            )
            if not health_result['all_passed']:
                # Fail fast with descriptive error
                error_msg = f"Transcription health checks failed: {health_result['message']}"
                logger.error(error_msg)
                # Log details for debugging
                for check in health_result.get('checks', []):
                    logger.error(f"  {check['name']}: {check['message']}")
                raise RuntimeError(error_msg)
            if show_progress:
                logger.info(f"Health checks passed: {health_result['message']}")

        # =========================================================================
        # US-124-012: FFmpeg Pipelining - Overlap extraction with transcription
        # =========================================================================
        if pipeline_depth > 0 and total_videos > 0:
            # Pipelined approach: Extract audio while transcribing previous videos
            # This overlaps I/O (extraction) with GPU computation (transcription)
            if show_progress:
                logger.info(f"PIPELINED mode: pipeline_depth={pipeline_depth}, max_workers={max_workers}, videos_to_process={total_videos}")

            audio_files = {}  # video_path -> audio_path
            phase1_start = time.time()
            phase2_start = time.time()
            phase1_time = 0.0
            phase2_time = 0.0

            # US-137-006: Pipeline efficiency tracking
            # Track extraction wait time (time waiting for audio to be ready) vs transcription time
            extraction_wait_times = []  # Time each video waited for extraction to complete
            transcription_times = []  # Time spent actually transcribing
            extraction_submission_times = {}  # video_path -> time when extraction was submitted
            pipeline_efficiency_samples = []  # Track efficiency over time

            # Create a thread pool for audio extraction
            extraction_executor = ThreadPoolExecutor(max_workers=max_workers)

            # Track extraction tasks
            pending_extractions = {}  # video_path -> future
            completed_extractions = {}  # video_path -> audio_path

            # Submit initial extractions based on pipeline depth
            next_extraction_idx = 0

            def submit_extraction(video_path):
                """Submit audio extraction task and return future."""
                # US-137-006: Track submission time for efficiency metrics
                extraction_submission_times[video_path] = time.time()
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                future = extraction_executor.submit(
                    extract_audio, video_path, str(temp_dir), audio_extraction_timeout
                )
                return future

            # Submit first batch of extractions
            initial_batch_size = min(pipeline_depth, total_videos)
            for i in range(initial_batch_size):
                video_path = uncached_videos[i]
                pending_extractions[video_path] = submit_extraction(video_path)
                next_extraction_idx = i + 1

            # Process videos in order, extracting more as we go
            transcribed_count = 0

            # Pre-compute audio durations for ETA (US-124-009)
            all_audio_durations = {}
            ROLLING_WINDOW_SIZE = 5
            recent_transcription_times = deque(maxlen=ROLLING_WINDOW_SIZE)

            while transcribed_count < total_videos:
                # Check for completed extractions
                newly_ready = []
                for video_path, future in list(pending_extractions.items()):
                    if future.done():
                        try:
                            audio_path = future.result()
                            if audio_path:
                                audio_files[video_path] = audio_path
                                completed_extractions[video_path] = audio_path

                                # US-137-006: Track extraction wait time
                                if video_path in extraction_submission_times:
                                    submit_time = extraction_submission_times[video_path]
                                    completion_time = time.time()
                                    wait_time = completion_time - submit_time
                                    extraction_wait_times.append(wait_time)
                                    del extraction_submission_times[video_path]
                            else:
                                completed_extractions[video_path] = None
                        except Exception as e:
                            log_error_with_context(logger, "TRANSCRIBE-001", f"Audio extraction error for {video_path}: {e}")
                            completed_extractions[video_path] = None
                        del pending_extractions[video_path]
                        newly_ready.append(video_path)

                # If we have ready audio, transcribe it
                # Process in order to maintain sequence
                ready_to_transcribe = [vp for vp in uncached_videos[:transcribed_count + len(completed_extractions)]
                                      if vp in completed_extractions and vp not in audio_files]

                for video_path in ready_to_transcribe:
                    if video_path not in audio_files:
                        continue

                    audio_path = audio_files[video_path]
                    if audio_path is None:
                        # Extraction failed, skip transcription
                        results[video_path] = []
                        metrics.record_failure(video_path)
                        transcribed_count += 1

                        # Clean up
                        try:
                            if audio_path and Path(audio_path).exists():
                                Path(audio_path).unlink()
                        except (OSError, IOError):
                            pass
                        continue

                    # Transcribe this video
                    video_name = Path(video_path).stem[:40]
                    overall_idx = transcribed_count

                    if show_progress and (overall_idx + 1) % progress_log_interval == 0 or overall_idx == 0 or overall_idx == total_videos - 1:
                        pct = ((overall_idx + 1) / total_videos) * 100
                        elapsed = time.time() - phase2_start
                        remaining = total_videos - overall_idx - 1
                        if overall_idx > 0 and len(recent_transcription_times) > 0:
                            avg_time = sum(recent_transcription_times) / len(recent_transcription_times)
                            eta = avg_time * remaining
                        else:
                            eta = (elapsed / (overall_idx + 1)) * remaining if overall_idx > 0 else 0
                        logger.info(f"[{overall_idx+1}/{total_videos}] {pct:.0f}% - {video_name} - ETA: {eta:.0f}s")

                    # Fire progress callback
                    pct = ((overall_idx + 1) / total_videos) * 100
                    is_interval = (overall_idx + 1) % callback_interval == 0
                    is_10_percent = pct > 0 and pct % 10 < (100 / total_videos)
                    if transcription_progress_callback and (is_interval or is_10_percent or overall_idx == total_videos - 1):
                        try:
                            transcription_progress_callback(pct, overall_idx + 1, total_videos, video_name)
                        except Exception as e:
                            logger.debug(f"Transcription progress callback failed: {e}")

                    # Check retry budget
                    if retry_budget.is_exhausted():
                        reason = retry_budget.exhaustion_reason()
                        logger.warning(f"TranscriptionRetryBudget: EXHAUSTED ({reason}), skipping {video_name}")
                        retry_budget.record_skip(video_path)
                        results[video_path] = []
                        metrics.record_failure(video_path)
                        transcribed_count += 1
                        continue

                    # Get audio duration for speed ratio
                    audio_duration = get_audio_duration(audio_path) or 0.0
                    all_audio_durations[video_path] = audio_duration

                    transcription_start = time.time()

                    # Retry loop
                    succeeded = False
                    for attempt in range(max_retries + 1):
                        retry_budget.record_attempt(video_path)

                        if attempt > 0 and retry_budget.is_exhausted():
                            reason = retry_budget.exhaustion_reason()
                            logger.warning(f"TranscriptionRetryBudget: EXHAUSTED during retries ({reason}), skipping")
                            retry_budget.record_skip(video_path)
                            break

                        try:
                            gpu_mem_before_mb, _ = _get_gpu_memory_mb()

                            transcription_result = whisper_client.transcribe(
                                audio_path,
                                language=language,
                                vad_filter=vad_filter,
                                min_silence_duration_ms=min_silence_duration_ms,
                                speech_pad_ms=speech_pad_ms
                            )
                            raw_segments, quality_metrics = transcription_result if isinstance(transcription_result, tuple) else (transcription_result, {})

                            gpu_mem_after_mb, _ = _get_gpu_memory_mb()

                            if quality_metrics:
                                metrics.record_confidence(
                                    video_path,
                                    quality_metrics.get('avg_word_confidence', 0.0),
                                    quality_metrics.get('min_segment_confidence', 1.0)
                                )

                            transcription_time = time.time() - transcription_start
                            recent_transcription_times.append(transcription_time)

                            # US-137-006: Track transcription time for efficiency metrics
                            transcription_times.append(transcription_time)

                            # US-137-006: Periodically adjust pipeline depth based on GPU utilization
                            if config and transcribed_count > 0 and transcribed_count % 5 == 0:
                                current_gpu_util = get_gpu_utilization()
                                pipeline_depth = adjust_pipeline_depth(pipeline_depth, current_gpu_util, config)

                            retry_budget.record_success(video_path)

                            # Cache result
                            transcript_cache.set(video_path, raw_segments)

                            results[video_path] = [
                                TranscriptSegment(
                                    index=j,
                                    start_time=seg['start'],
                                    end_time=seg['end'],
                                    text=seg['text'],
                                    source_file=video_path
                                )
                                for j, seg in enumerate(raw_segments)
                            ]

                            metrics.record_transcription(video_path, audio_duration, transcription_time)

                            # Record backoff success for adaptive strategy (US-137-011)
                            if attempt > 0:  # Only if we actually used backoff
                                backoff_manager.record_success()

                            succeeded = True
                            break

                        except Exception as e:
                            retry_budget.record_failure(video_path)
                            if not is_transient_error(e):
                                log_error_with_context(logger, "TRANSCRIBE-001", f"Transcription failed for {video_name} (permanent): {e}")

                                # Record backoff failure for adaptive strategy (US-137-011)
                                if attempt > 0:
                                    backoff_manager.record_failure()

                                break
                            if attempt < max_retries:
                                # Use backoff manager with jitter correlation (US-137-011)
                                delay = backoff_manager.calculate_delay(attempt, worker_id=None)
                                retry_budget.record_backoff(delay)
                                logger.warning(f"Retrying {video_name} (attempt {attempt + 1}/{max_retries + 1}, delay={delay:.2f}s): {e}")
                                _clear_cuda_cache()
                                time.sleep(delay)
                            else:
                                logger.error(f"Transcription failed for {video_name} after {max_retries + 1} attempts: {e}")

                    if not succeeded:
                        results[video_path] = []
                        if video_path not in [v for v in retry_budget.skipped_video_ids]:
                            metrics.record_failure(video_path)

                    # Clean up audio file
                    try:
                        Path(audio_path).unlink()
                    except (OSError, IOError):
                        pass

                    transcribed_count += 1

                    # Submit more extractions if available
                    while next_extraction_idx < total_videos and len(pending_extractions) < pipeline_depth:
                        video_path = uncached_videos[next_extraction_idx]
                        pending_extractions[video_path] = submit_extraction(video_path)
                        next_extraction_idx += 1

                # Small sleep to avoid busy-waiting
                if not ready_to_transcribe:
                    time.sleep(0.1)

            # Cleanup extraction executor
            extraction_executor.shutdown(wait=False)

            phase1_time = time.time() - phase1_start
            phase2_time = time.time() - phase2_start

            if show_progress:
                logger.info(f"PIPELINED complete: {transcribed_count} videos ({phase1_time:.1f}s extraction, {phase2_time:.1f}s transcription)")

            # US-137-006: Log pipeline efficiency metrics
            if extraction_wait_times and transcription_times:
                avg_wait = sum(extraction_wait_times) / len(extraction_wait_times)
                avg_transcribe = sum(transcription_times) / len(transcription_times)
                # Efficiency: ratio of wait time to transcription time
                # Lower ratio = more efficient (less waiting relative to work)
                efficiency_ratio = avg_wait / avg_transcribe if avg_transcribe > 0 else 0
                # Calculate overlap: how much extraction happened during transcription
                # If extraction wait time < transcription time, we have good overlap
                total_extraction_time = sum(extraction_wait_times)
                total_transcribe_time = sum(transcription_times)
                overlap_percent = min(100, (total_extraction_time / total_transcribe_time * 100) if total_transcribe_time > 0 else 0)

                logger.info(
                    f"Pipeline efficiency (US-137-006): "
                    f"avg_wait={avg_wait:.2f}s, avg_transcribe={avg_transcribe:.2f}s, "
                    f"efficiency_ratio={efficiency_ratio:.2f}, overlap={overlap_percent:.1f}%"
                )

                # Record in metrics if available
                if hasattr(metrics, 'set_pipeline_efficiency'):
                    metrics.set_pipeline_efficiency(avg_wait, avg_transcribe, efficiency_ratio)

            # Record phase times in metrics
            metrics.set_phase_times(phase1_time, phase2_time)

        else:
            # Traditional two-phase approach (pipeline_depth == 0 or no videos)
            # =========================================================================
            # PHASE 1: Parallel audio extraction (CPU-bound) - with batching (US-110-005)
            # =========================================================================
            if show_progress:
                logger.info(f"Phase 1: Extracting audio ({max_workers} workers)...")

            audio_files = {}
            phase1_start = time.time()

            # US-110-005: Prepare batches of videos
            if video_batch_size > 0:
                batches = [uncached_videos[i:i + video_batch_size] for i in range(0, len(uncached_videos), video_batch_size)]
                num_batches = len(batches)
            else:
                batches = [uncached_videos]
                num_batches = 1

            def extract_audio_task(video_path):
                audio_path = extract_audio(video_path, str(temp_dir), timeout=audio_extraction_timeout)
                return video_path, audio_path

            batch_completed = 0

            for batch_idx, batch_videos in enumerate(batches):
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(extract_audio_task, vp)
                        for vp in batch_videos
                    ]

                    completed = 0
                    batch_total = len(futures)
                    for future in as_completed(futures):
                        try:
                            video_path, audio_path = future.result()
                            completed += 1
                            batch_completed += 1
                            if audio_path:
                                audio_files[video_path] = audio_path
                            if show_progress and completed % progress_log_interval == 0:
                                logger.info(f"Batch {batch_idx + 1}/{num_batches}: Extracted {completed}/{batch_total} audio files")
                        except Exception as e:
                            completed += 1
                            batch_completed += 1
                            log_error_with_context(logger, "TRANSCRIBE-001", f"Audio extraction error: {e}")

                if num_batches > 1 and batch_idx < num_batches - 1 and batch_wait_seconds > 0:
                    if show_progress:
                        logger.info(f"Batch {batch_idx + 1}/{num_batches} complete, waiting {batch_wait_seconds}s...")
                    time.sleep(batch_wait_seconds)

            phase1_time = time.time() - phase1_start
            if show_progress:
                logger.info(f"Phase 1 complete: {len(audio_files)} videos ready ({phase1_time:.1f}s)")

            # =========================================================================
            # PHASE 2: Sequential GPU transcription (mutex protected) - with batching (US-110-005)
            # =========================================================================
            if show_progress:
                logger.info("Phase 2: Transcribing with shared model (sequential GPU)...")

            phase2_start = time.time()
            total = len(audio_files)

        # US-110-005: Prepare batches for Phase 2
        audio_files_list = list(audio_files.items())
        if video_batch_size > 0:
            phase2_batches = [audio_files_list[i:i + video_batch_size] for i in range(0, len(audio_files_list), video_batch_size)]
            num_phase2_batches = len(phase2_batches)
        else:
            phase2_batches = [audio_files_list]
            num_phase2_batches = 1

        # US-110-005: Log Phase 2 batch configuration
        if show_progress and video_batch_size > 0 and total > video_batch_size:
            logger.info(f"Phase 2: {num_phase2_batches} batches of up to {video_batch_size} videos")

        batch_processed = 0

        # US-124-009: Pre-compute audio durations for all videos for accurate ETA
        # This allows ETA calculation to account for segment length variance
        all_audio_durations = {}
        for video_path, audio_path in audio_files_list:
            duration = get_audio_duration(audio_path) or 0.0
            all_audio_durations[video_path] = duration

        # US-124-009: Rolling average ETA tracking (last N videos)
        ROLLING_WINDOW_SIZE = 5
        recent_transcription_times = deque(maxlen=ROLLING_WINDOW_SIZE)
        completed_audio_durations = {}  # Track completed videos' durations

        for batch_idx, batch_audio_files in enumerate(phase2_batches):
            # Process current batch
            for i, (video_path, audio_path) in enumerate(batch_audio_files):
                overall_idx = batch_processed + i
                video_name = Path(video_path).stem[:40]

                if show_progress and (overall_idx + 1) % progress_log_interval == 0 or overall_idx == 0 or overall_idx == total - 1:
                    pct = ((overall_idx + 1) / total) * 100
                    elapsed = time.time() - phase2_start

                    # US-124-009: Rolling average + segment length variance ETA
                    remaining_count = total - overall_idx - 1
                    if overall_idx > 0 and len(recent_transcription_times) > 0:
                        # Use rolling average of recent transcription times
                        avg_recent_time = sum(recent_transcription_times) / len(recent_transcription_times)

                        # Account for segment length variance: weight by remaining audio durations
                        remaining_audio_durations = [
                            all_audio_durations[vp] for vp, ap in audio_files_list[overall_idx + 1:]
                        ]
                        completed_audio_durs = list(completed_audio_durations.values())

                        if completed_audio_durs and sum(remaining_audio_durations) > 0:
                            avg_completed_duration = sum(completed_audio_durs) / len(completed_audio_durs)
                            avg_remaining_duration = sum(remaining_audio_durations) / len(remaining_audio_durations)
                            # Variance factor: if remaining videos are longer, increase ETA
                            audio_variance_factor = avg_remaining_duration / avg_completed_duration if avg_completed_duration > 0 else 1.0
                        else:
                            audio_variance_factor = 1.0

                        eta = avg_recent_time * remaining_count * audio_variance_factor
                    else:
                        # Fallback to simple average for first video
                        eta = (elapsed / (overall_idx + 1)) * remaining_count if overall_idx > 0 else 0

                    logger.info(f"[{overall_idx+1}/{total}] {pct:.0f}% - {video_name} - ETA: {eta:.0f}s")

                # US-110-002: Fire transcription progress callback at configurable intervals
                # Callback fires every callback_interval videos OR at 10% progress milestones
                pct = ((overall_idx + 1) / total) * 100
                is_interval = (overall_idx + 1) % callback_interval == 0
                is_10_percent = pct > 0 and pct % 10 < (100 / total)
                if transcription_progress_callback and (is_interval or is_10_percent or overall_idx == total - 1):
                    try:
                        transcription_progress_callback(pct, overall_idx + 1, total, video_name)
                    except Exception as e:
                        logger.debug(f"Transcription progress callback failed: {e}")
            video_name = Path(video_path).stem[:40]

            if show_progress and (i + 1) % progress_log_interval == 0 or i == 0 or i == total - 1:
                pct = ((i + 1) / total) * 100
                elapsed = time.time() - phase2_start

                # US-124-009: Rolling average + segment length variance ETA (batch-level)
                remaining_count = total - i - 1
                current_idx = batch_processed + i
                if current_idx > 0 and len(recent_transcription_times) > 0:
                    avg_recent_time = sum(recent_transcription_times) / len(recent_transcription_times)

                    remaining_audio_durations = [
                        all_audio_durations[vp] for vp, ap in audio_files_list[current_idx + 1:]
                    ]
                    completed_audio_durs = list(completed_audio_durations.values())

                    if completed_audio_durs and sum(remaining_audio_durations) > 0:
                        avg_completed_duration = sum(completed_audio_durs) / len(completed_audio_durs)
                        avg_remaining_duration = sum(remaining_audio_durations) / len(remaining_audio_durations)
                        audio_variance_factor = avg_remaining_duration / avg_completed_duration if avg_completed_duration > 0 else 1.0
                    else:
                        audio_variance_factor = 1.0

                    eta = avg_recent_time * remaining_count * audio_variance_factor
                else:
                    eta = (elapsed / (current_idx + 1)) * remaining_count if current_idx > 0 else 0

                logger.info(f"[{i+1}/{total}] {pct:.0f}% - {video_name} - ETA: {eta:.0f}s")

            # US-110-002: Fire transcription progress callback at configurable intervals
            # Callback fires every callback_interval videos OR at 10% progress milestones
            pct = ((i + 1) / total) * 100
            is_interval = (i + 1) % callback_interval == 0
            is_10_percent = pct > 0 and pct % 10 < (100 / total)
            if transcription_progress_callback and (is_interval or is_10_percent or i == total - 1):
                try:
                    transcription_progress_callback(pct, i + 1, total, video_name)
                except Exception as e:
                    logger.debug(f"Transcription progress callback failed: {e}")

            # Check retry budget before attempting (US-79-010)
            if retry_budget.is_exhausted():
                reason = retry_budget.exhaustion_reason()
                logger.warning(
                    f"TranscriptionRetryBudget: EXHAUSTED ({reason}), "
                    f"skipping {video_name} and remaining videos"
                )
                retry_budget.record_skip(video_path)
                results[video_path] = []
                metrics.record_failure(video_path)
                continue

            # Get audio duration for speed ratio calculation (US-60-009)
            audio_duration = get_audio_duration(audio_path) or 0.0
            transcription_start = time.time()

            # Retry loop with budget tracking (US-79-010)
            succeeded = False
            for attempt in range(max_retries + 1):
                retry_budget.record_attempt(video_path)

                # Re-check budget before retry attempts (not first attempt)
                if attempt > 0 and retry_budget.is_exhausted():
                    reason = retry_budget.exhaustion_reason()
                    logger.warning(
                        f"TranscriptionRetryBudget: EXHAUSTED during retries for {video_name} "
                        f"({reason}), skipping"
                    )
                    retry_budget.record_skip(video_path)
                    break

                try:
                    # US-110-006: Track GPU memory before and after transcription
                    gpu_mem_before_mb = 0.0
                    gpu_mem_after_mb = 0.0

                    # Get GPU memory before transcription
                    gpu_mem_before_mb, _ = _get_gpu_memory_mb()

                    # Transcribe with WhisperClient (GPU-locked) - returns (segments, quality_metrics)
                    transcription_result = whisper_client.transcribe(
                        audio_path,
                        language=language,
                        vad_filter=vad_filter,
                        min_silence_duration_ms=min_silence_duration_ms,
                        speech_pad_ms=speech_pad_ms
                    )
                    raw_segments, quality_metrics = transcription_result if isinstance(transcription_result, tuple) else (transcription_result, {})

                    # Log batch transcription segment boundaries and count
                    if raw_segments:
                        segment_count = len(raw_segments)
                        if raw_segments:
                            first_start = raw_segments[0].get('start', 0)
                            last_end = raw_segments[-1].get('end', 0)
                            total_audio_duration = last_end - first_start
                        else:
                            first_start = last_end = total_audio_duration = 0
                        logger.info(
                            f"[TRANSCRIBE] Batch transcription: {video_name}, "
                            f"segments={segment_count}, time_range=({first_start:.2f}s - {last_end:.2f}s), "
                            f"total_duration={total_audio_duration:.2f}s"
                        )

                    # Get GPU memory after transcription
                    gpu_mem_after_mb, _ = _get_gpu_memory_mb()

                    # Calculate memory delta (allocated during transcription)
                    gpu_mem_delta_mb = max(0, gpu_mem_after_mb - gpu_mem_before_mb)

                    # Log memory delta for this video
                    if gpu_mem_delta_mb > 0:
                        logger.debug(f"GPU memory delta for {video_name}: {gpu_mem_delta_mb:.1f} MB")

                    # Check if GPU memory usage exceeds 90% (US-110-006)
                    if gpu_mem_after_mb > 0:
                        available_gpu = get_available_gpu_memory()
                        if available_gpu > 0:
                            total_gpu = gpu_mem_after_mb + available_gpu
                            usage_percent = (gpu_mem_after_mb / total_gpu) * 100
                            if usage_percent > 90:
                                logger.warning(
                                    f"GPU memory usage high: {usage_percent:.1f}% "
                                    f"({gpu_mem_after_mb:.0f} MB / {total_gpu:.0f} MB) "
                                    f"for {video_name}"
                                )

                    # Record quality metrics (US-110-007)
                    if quality_metrics:
                        metrics.record_confidence(
                            video_path,
                            quality_metrics.get('avg_word_confidence', 0.0),
                            quality_metrics.get('min_segment_confidence', 1.0)
                        )

                    transcription_time = time.time() - transcription_start

                    # US-124-009: Track transcription time for rolling average ETA
                    recent_transcription_times.append(transcription_time)
                    completed_audio_durations[video_path] = audio_duration

                    retry_budget.record_success(video_path)

                    # Cache the result
                    transcript_cache.set(video_path, raw_segments)

                    # Convert to TranscriptSegment
                    results[video_path] = [
                        TranscriptSegment(
                            index=j,
                            start_time=seg['start'],
                            end_time=seg['end'],
                            text=seg['text'],
                            source_file=video_path
                        )
                        for j, seg in enumerate(raw_segments)
                    ]

                    # Record metrics (US-60-009)
                    metrics.record_transcription(video_path, audio_duration, transcription_time)

                    # Record GPU memory usage (US-110-006)
                    if gpu_mem_delta_mb > 0:
                        metrics.record_gpu_memory(video_path, gpu_mem_delta_mb)

                    # Record backoff success for adaptive strategy (US-137-011)
                    if attempt > 0:  # Only if we actually used backoff
                        backoff_manager.record_success()

                    succeeded = True
                    break  # Success, exit retry loop

                except Exception as e:
                    retry_budget.record_failure(video_path)

                    # Check if error is transient (worth retrying)
                    if not is_transient_error(e):
                        log_error_with_context(logger, "TRANSCRIBE-001", f"Transcription failed for {video_name} (permanent error): {e}")
                        break

                    # Transient error — retry with backoff if attempts remain
                    if attempt < max_retries:
                        # Use backoff manager with jitter correlation (US-137-011)
                        delay = backoff_manager.calculate_delay(attempt, worker_id=None)
                        retry_budget.record_backoff(delay)
                        logger.warning(
                            f"Retrying transcription for {video_name} "
                            f"(attempt {attempt + 1}/{max_retries + 1}, delay={delay:.2f}s): {e}"
                        )
                        _clear_cuda_cache()
                        time.sleep(delay)
                    else:
                        # Record backoff failure for adaptive strategy (US-137-011)
                        if attempt > 0:  # Only if we actually used backoff
                            backoff_manager.record_failure()

                        logger.error(
                            f"  Transcription failed for {video_name} "
                            f"after {max_retries + 1} attempts: {e}"
                        )

            if not succeeded:
                results[video_path] = []
                # Only record failure if not already recorded via budget skip
                if video_path not in [v for v in retry_budget.skipped_video_ids]:
                    metrics.record_failure(video_path)

            # Clean up audio file
            try:
                Path(audio_path).unlink()
            except (OSError, IOError) as e:
                # Non-critical: temp file cleanup failure won't affect results
                logger.debug(f"Could not remove temp audio file {audio_path}: {e}")

            batch_processed += 1

        # US-110-005: Wait between Phase 2 batches (except after last batch)
        if num_phase2_batches > 1 and batch_idx < num_phase2_batches - 1 and batch_wait_seconds > 0:
            if show_progress:
                logger.info(f"Phase 2 batch {batch_idx + 1}/{num_phase2_batches} complete, waiting {batch_wait_seconds}s before next batch...")
            # Clean up GPU memory between batches
            if auto_cleanup_after_batch:
                whisper_client.cleanup()
                # Reinitialize for next batch
                whisper_client = WhisperClient(
                    model_name=model_name, model_version=model_version, compute_type=compute_type,
                    gpu_transcription_timeout=gpu_transcription_timeout,
                    num_workers=whisper_num_workers,
                    cpu_threads=whisper_cpu_threads,
                    auto_fallback_to_cpu=auto_fallback_to_cpu,
                    auto_model_selection=auto_model_selection  # US-110-010
                )
            time.sleep(batch_wait_seconds)

        phase2_time = time.time() - phase2_start
        if show_progress:
            logger.info(f"Phase 2 complete: {len(results)} videos ({phase2_time:.1f}s)")

        # Record phase times in metrics (US-60-009)
        metrics.set_phase_times(phase1_time, phase2_time)

        # Record retry budget summary in metrics (US-79-010)
        budget_summary = retry_budget.get_summary()
        metrics.set_retry_budget_summary(budget_summary)
        if budget_summary['is_exhausted']:
            logger.warning(
                f"TranscriptionRetryBudget exhausted: {budget_summary['exhaustion_reason']}. "
                f"Skipped {budget_summary['videos_skipped']} videos."
            )

        # Log backoff strategy stats (US-137-011)
        backoff_stats = backoff_manager.get_strategy_stats()
        if show_progress:
            effective = backoff_manager.get_best_strategy()
            logger.info(
                f"Backoff strategy: {backoff_manager.strategy.value} "
                f"(effective: {effective.value}, success rate: {backoff_manager.get_success_rate():.1%})"
            )

        # Log metrics summary at end of batch transcription (US-60-009)
        summary = metrics.get_summary_dict()
        logger.info(
            f"Transcription batch complete: {summary['transcribed_count']} transcribed, "
            f"{summary['cached_hits']} cached, {summary['failed_count']} failed. "
            f"Speed: {summary['avg_speed_ratio']:.1f}x realtime"
        )
        if show_progress and summary['transcribed_count'] > 0:
            logger.info(
                f"Speed: {summary['avg_speed_ratio']:.1f}x realtime "
                f"({summary['total_duration_s']:.0f}s audio in {summary['total_time_s']:.0f}s)"
            )

        # US-79-012: Persist transcription metrics to checkpoint for cross-run comparison
        if checkpoint is not None:
            try:
                # Build summary with acceptance-criteria field names
                checkpoint_summary = {
                    'total_videos': summary['total_videos'],
                    'cached_videos': summary['cached_hits'],
                    'transcribed_videos': summary['transcribed_count'],
                    'failed_videos': summary['failed_count'],
                    'total_duration_seconds': summary['total_duration_s'],
                    'average_speed_ratio': summary['avg_speed_ratio'],
                    # Also include detailed fields for debugging
                    'cache_hit_rate': summary['cache_hit_rate'],
                    'success_rate': summary['success_rate'],
                    'phase1_time_s': summary['phase1_time_s'],
                    'phase2_time_s': summary['phase2_time_s'],
                    'budget_total_attempts': summary.get('budget_total_attempts', 0),
                    'budget_failed_attempts': summary.get('budget_failed_attempts', 0),
                    'budget_exhausted_count': summary.get('budget_exhausted_count', 0),
                }
                checkpoint.save_transcription_metrics(checkpoint_summary)
            except Exception as e:
                logger.warning(f"Could not persist transcription metrics to checkpoint: {e}")

        # US-137-003: Export metrics in multiple formats after batch completion
        try:
            metrics_export_path = getattr(
                getattr(config, 'transcription', None), 'metrics_export_path', ''
            ) or ''
            metrics_export_formats = getattr(
                getattr(config, 'transcription', None), 'metrics_export_formats', 'json'
            ) or 'json'

            if metrics_export_path:
                from datetime import datetime
                export_dir = Path(metrics_export_path)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                formats = [f.strip().lower() for f in metrics_export_formats.split(',')]

                if 'json' in formats:
                    json_path = export_dir / f"transcription_metrics_{timestamp}.json"
                    metrics.export_json(json_path)
                    logger.info(f"Exported transcription metrics to JSON: {json_path}")

                if 'prometheus' in formats:
                    prom_path = export_dir / f"transcription_metrics_{timestamp}.prom"
                    metrics.export_prometheus(prom_path)
                    logger.info(f"Exported transcription metrics to Prometheus: {prom_path}")

                if 'csv' in formats:
                    csv_path = export_dir / f"transcription_metrics_{timestamp}.csv"
                    metrics.export_csv(csv_path)
                    logger.info(f"Exported transcription metrics to CSV: {csv_path}")
        except Exception as e:
            logger.warning(f"Could not export transcription metrics: {e}")

        # Clean up temp directory
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except (OSError, IOError) as e:
            # Non-critical: temp directory cleanup failure won't affect results
            logger.debug(f"Could not remove temp directory {temp_dir}: {e}")

        if return_metrics:
            return results, metrics
        return results

    finally:
        # Cleanup WhisperModel to free GPU memory (US-60-011)
        # Called even if batch processing raises an exception
        if auto_cleanup_after_batch:
            logger.info("Auto-cleanup after batch enabled, freeing GPU memory...")
            whisper_client.cleanup()


def transcribe_video(
    video_path: str,
    cache: TranscriptCache,
    model_name: str = "base",
    compute_type: str = "auto",
    language: str = None,
    temp_dir: str = None,
    vad_filter: bool = False,  # Default False - VAD too aggressive for YouTube
    min_silence_duration_ms: int = 200,
    speech_pad_ms: int = 10,
    max_retries: int = 2,
    base_delay: float = 1.0,
    gpu_transcription_timeout: int = 300,
    audio_extraction_timeout: int = 60,
    num_workers: int = 1,
    cpu_threads: int = 4,
    auto_fallback_to_cpu: bool = True,  # US-110-004
    auto_model_selection: bool = True  # US-110-010
) -> List[TranscriptSegment]:
    """
    Transcribe a single video file with retry logic for transient errors.

    Uses cache if available, otherwise transcribes with shared model.
    Retries up to max_retries times with exponential backoff on transient errors
    (e.g., GPU memory pressure). Does NOT retry on file not found or permission errors.

    Args:
        video_path: Path to video file
        cache: TranscriptCache instance
        model_name: Whisper model name
        compute_type: Compute type (auto, float16, int8)
        language: Language code or None for auto-detect
        temp_dir: Temporary directory for audio extraction
        vad_filter: Whether to apply Voice Activity Detection
        min_silence_duration_ms: Minimum silence duration to split segments
        speech_pad_ms: Padding around detected speech
        max_retries: Maximum retry attempts for transient errors (default 2)
        base_delay: Base delay for exponential backoff in seconds (default 1.0)
        gpu_transcription_timeout: Max seconds for a single transcribe() call (US-79-002)
        audio_extraction_timeout: Max seconds for FFmpeg audio extraction (US-79-003)
    """
    video_path = str(video_path)
    video_name = Path(video_path).name
    video_id = Path(video_path).stem  # For logging

    # Check cache first
    cached = cache.get(video_path)
    if cached:
        return [
            TranscriptSegment(
                index=i,
                start_time=seg['start'],
                end_time=seg['end'],
                text=seg['text'],
                source_file=video_path
            )
            for i, seg in enumerate(cached)
        ]

    # Extract audio
    audio_path = extract_audio(video_path, temp_dir, timeout=audio_extraction_timeout)
    if not audio_path:
        logger.warning(f"  Could not extract audio: {video_name}")
        return []

    # Transcribe with WhisperClient (GPU-locked) - with retry for transient errors
    whisper_client = WhisperClient(
        model_name=model_name, model_version=model_version, compute_type=compute_type,
        gpu_transcription_timeout=gpu_transcription_timeout,
        num_workers=num_workers, cpu_threads=cpu_threads,
        auto_fallback_to_cpu=auto_fallback_to_cpu,  # US-110-004
        auto_model_selection=auto_model_selection  # US-110-010
    )
    raw_segments = None
    last_error = None

    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            transcription_result = whisper_client.transcribe(
                audio_path,
                language=language,
                vad_filter=vad_filter,
                min_silence_duration_ms=min_silence_duration_ms,
                speech_pad_ms=speech_pad_ms
            )
            raw_segments = transcription_result if isinstance(transcription_result, list) else transcription_result[0]
            break  # Success, exit retry loop

        except Exception as e:
            last_error = e

            # Check if error is transient (worth retrying)
            if not is_transient_error(e):
                # Permanent error - don't retry
                log_error_with_context(logger, "TRANSCRIBE-001", f"Transcription failed for {video_name} (permanent error): {e}")
                # Clean up audio file
                try:
                    Path(audio_path).unlink()
                except (OSError, IOError):
                    pass
                return []

            # Transient error - retry if attempts remain
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)  # Exponential backoff: 1s, 2s
                logger.warning(
                    f"Retrying transcription for {video_id} after {e}"
                )
                # Clear CUDA cache to help recover from GPU OOM (US-60-012)
                _clear_cuda_cache()
                time.sleep(delay)
            else:
                # All retries exhausted
                logger.error(
                    f"  Transcription failed for {video_name} after {max_retries + 1} attempts: {e}"
                )

    # Process result if we have segments
    if raw_segments is not None:
        # Cache the result
        cache.set(video_path, raw_segments)

        # Clean up audio file
        try:
            Path(audio_path).unlink()
        except (OSError, IOError) as e:
            # Non-critical: temp file cleanup failure won't affect results
            logger.debug(f"Could not remove temp audio file {audio_path}: {e}")

        return [
            TranscriptSegment(
                index=i,
                start_time=seg['start'],
                end_time=seg['end'],
                text=seg['text'],
                source_file=video_path
            )
            for i, seg in enumerate(raw_segments)
        ]

    # Clean up audio file on failure
    try:
        Path(audio_path).unlink()
    except (OSError, IOError):
        pass

    return []


def transcribe_voiceover_audio(
    audio_path: str,
    model_name: str = "base",
    model_version: str = None,  # US-124-010
    compute_type: str = "auto",
    language: str = None,
    vad_filter: bool = True,  # Enable VAD by default for voiceover - better gap detection
    gpu_transcription_timeout: int = 300,
    num_workers: int = 1,
    cpu_threads: int = 4,
    auto_fallback_to_cpu: bool = True,  # US-110-004
    auto_model_selection: bool = True  # US-110-010
) -> List[dict]:
    """
    Transcribe a voiceover audio file.
    Backward-compatible function for voiceover transcription.

    Args:
        vad_filter: Whether to apply Voice Activity Detection. Default True for voiceover
                   as it produces cleaner segment boundaries with accurate gap timing.
        gpu_transcription_timeout: Max seconds for a single transcribe() call (US-79-002)
    """
    whisper_client = WhisperClient(
        model_name=model_name, model_version=model_version, compute_type=compute_type,
        gpu_transcription_timeout=gpu_transcription_timeout,
        num_workers=num_workers, cpu_threads=cpu_threads,
        auto_fallback_to_cpu=auto_fallback_to_cpu,  # US-110-004
        auto_model_selection=auto_model_selection  # US-110-010
    )
    logger.info(f"Transcribing voiceover with VAD={'enabled' if vad_filter else 'disabled'}")
    result = whisper_client.transcribe(
        audio_path,
        language=language,
        vad_filter=vad_filter
    )
    # Return just segments (tuple unpacked by caller if needed)
    return result if isinstance(result, list) else result[0]


def transcribe_voiceover_media(
    media_path: str,
    output_srt_path: str = None,
    model_name: str = "base",
    model_version: str = None,  # US-124-010
    language: str = None,
    compute_type: str = "auto",
    cache_dir: str = None,
    word_timestamps: bool = True,
    vad_filter: bool = True,  # Enable VAD by default for voiceover - better gap detection
    gpu_transcription_timeout: int = 300,
    audio_extraction_timeout: int = 60,
    num_workers: int = 1,
    cpu_threads: int = 4,
    auto_fallback_to_cpu: bool = True,  # US-110-004
    auto_model_selection: bool = True  # US-110-010
) -> str:
    """
    Transcribe voiceover from any media file (audio or video) and save as SRT.

    Args:
        media_path: Path to audio or video file
        output_srt_path: Path for output SRT file (default: same as media with .srt extension)
        model_name: Whisper model name
        language: Language code or None for auto-detect
        compute_type: Compute type (auto, float16, int8)
        cache_dir: Optional cache directory for extracted audio
        word_timestamps: Whether to generate word-level timestamps (default True)
        vad_filter: Whether to apply Voice Activity Detection. Default True for voiceover
                   as it produces cleaner segment boundaries with accurate gap timing.

    Returns:
        Path to the generated SRT file (also generates .words.json if word_timestamps=True)
    """
    media_path = Path(media_path)

    # Determine output path
    if output_srt_path:
        srt_path = Path(output_srt_path)
    else:
        srt_path = media_path.with_suffix('.srt')

    # Check if it's a video file that needs audio extraction
    video_extensions = {'.mp4', '.mkv', '.webm', '.avi', '.mov', '.mxf'}
    audio_extensions = {'.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac'}

    segments = []
    whisper_client = WhisperClient(
        model_name=model_name, model_version=model_version, compute_type=compute_type,
        gpu_transcription_timeout=gpu_transcription_timeout,
        num_workers=num_workers, cpu_threads=cpu_threads,
        auto_fallback_to_cpu=auto_fallback_to_cpu,  # US-110-004
        auto_model_selection=auto_model_selection  # US-110-010
    )

    if media_path.suffix.lower() in video_extensions:
        # Extract audio first
        if cache_dir:
            temp_dir = Path(cache_dir) / "temp_audio"
            temp_dir.mkdir(parents=True, exist_ok=True)
        else:
            temp_dir = media_path.parent

        audio_path = extract_audio(str(media_path), str(temp_dir), timeout=audio_extraction_timeout)

        if not audio_path:
            logger.error(f"Could not extract audio from {media_path}")
            raise RuntimeError(f"Could not extract audio from {media_path}")

        # Transcribe the extracted audio with word timestamps
        logger.info(f"Transcribing voiceover with VAD={'enabled' if vad_filter else 'disabled'}")
        result = whisper_client.transcribe(
            audio_path,
            language=language,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps
        )
        segments = result if isinstance(result, list) else result[0]

        # Clean up extracted audio
        try:
            Path(audio_path).unlink()
        except (OSError, IOError) as e:
            # Non-critical: temp file cleanup failure won't affect results
            logger.debug(f"Could not remove extracted audio file {audio_path}: {e}")

    elif media_path.suffix.lower() in audio_extensions:
        # It's already an audio file
        logger.info(f"Transcribing voiceover with VAD={'enabled' if vad_filter else 'disabled'}")
        result = whisper_client.transcribe(
            str(media_path),
            language=language,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps
        )
        segments = result if isinstance(result, list) else result[0]
    else:
        raise ValueError(f"Unsupported media format: {media_path.suffix}")

    if not segments:
        raise RuntimeError(f"No segments generated from transcription of {media_path}")

    # Write SRT file
    write_srt(segments, str(srt_path))

    # Save word-level timestamps to JSON for pause-split accuracy
    if word_timestamps:
        words_path = srt_path.with_suffix('.words.json')
        try:
            with open(words_path, 'w', encoding='utf-8') as f:
                json.dump(segments, f, indent=2)
            logger.info(f"Saved word timestamps to {words_path}")
        except Exception as e:
            logger.warning(f"Could not save word timestamps: {e}")

    return str(srt_path)


def get_transcript_segments(
    video_path: str,
    cache_dir: str,
    model_name: str = "base",
    compute_type: str = "auto",
    compress_cache: bool = True  # US-110-008
) -> List[TranscriptSegment]:
    """
    Get transcript segments for a single video.
    Backward-compatible function.
    """
    cache = TranscriptCache(cache_dir, compress_cache=compress_cache, min_segment_words=3)  # US-110-009
    return transcribe_video(video_path, cache, model_name, compute_type)
