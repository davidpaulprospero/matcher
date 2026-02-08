"""
Parallel video transcription processor.

Orchestrates batch transcription with two-phase processing:
1. Parallel audio extraction (CPU-bound, I/O)
2. Sequential GPU transcription (shared WhisperModel)

Also provides single-video and voiceover transcription wrappers.
"""

import json
import logging
import shutil
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from .whisper_client import WhisperClient
from .cache import TranscriptCache
from .utils import extract_audio, write_srt, get_audio_duration
from .exceptions import is_transient_error
from .metrics import TranscriptionMetrics
from src.state import TranscriptSegment

logger = logging.getLogger(__name__)


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


def transcribe_videos_parallel(
    video_paths: List[str],
    cache: Any,
    config: Any = None,
    max_workers: int = None,
    force_reprocess: bool = False,
    show_progress: bool = True,
    skip_if_cached: bool = True,
    return_metrics: bool = False
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
        compute_type = getattr(config.transcription, 'compute_type', 'auto')
        language = getattr(config.transcription, 'language', None)
        # VAD is ALWAYS disabled for video transcription - config setting is for voiceover only
        # YouTube video audio quality varies, VAD is too aggressive and removes speech
        vad_filter = False
        min_silence_duration_ms = getattr(config.transcription, 'min_silence_duration_ms', 200)
        speech_pad_ms = getattr(config.transcription, 'speech_pad_ms', 10)
        # Audio extraction workers (US-60-010)
        if max_workers is None:
            max_workers = getattr(config.transcription, 'audio_extraction_workers', 4)
    else:
        model_name = "base"
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

    # Get progress logging interval (US-79-008)
    progress_log_interval = 10  # Default every 10 items
    if config:
        progress_log_interval = getattr(config.transcription, 'progress_log_interval', 10)

    # Initialize WhisperClient and TranscriptCache
    whisper_client = WhisperClient(
        model_name=model_name, compute_type=compute_type,
        gpu_transcription_timeout=gpu_transcription_timeout,
        num_workers=whisper_num_workers,
        cpu_threads=whisper_cpu_threads
    )
    transcript_cache = TranscriptCache(cache_dir)
    results = {}

    try:
        # Initialize metrics (US-60-009)
        metrics = TranscriptionMetrics(total_videos=len(video_paths))

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

        if not uncached_videos:
            # Log metrics summary even when all cached (US-60-009)
            logger.info(f"Transcription metrics: {metrics.get_summary_dict()}")
            if return_metrics:
                return results, metrics
            return results

        # Create temp directory for audio files
        temp_dir = Path(cache_dir) / "temp_audio"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # =========================================================================
        # PHASE 1: Parallel audio extraction (CPU-bound)
        # =========================================================================
        if show_progress:
            logger.info(f"Phase 1: Extracting audio ({max_workers} workers)...")

        audio_files = {}  # video_path -> audio_path
        phase1_start = time.time()

        def extract_audio_task(video_path):
            audio_path = extract_audio(video_path, str(temp_dir), timeout=audio_extraction_timeout)
            return video_path, audio_path

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(extract_audio_task, vp)
                for vp in uncached_videos
            ]

            completed = 0
            total_videos = len(futures)
            for future in as_completed(futures):
                try:
                    video_path, audio_path = future.result()
                    completed += 1
                    if audio_path:
                        audio_files[video_path] = audio_path
                    if show_progress and completed % progress_log_interval == 0:
                        logger.info(f"Extracted {completed}/{total_videos} audio files")
                except Exception as e:
                    completed += 1
                    # Log full traceback for debugging parallel processing issues
                    logger.exception(f"  Audio extraction error: {e}")

        phase1_time = time.time() - phase1_start
        if show_progress:
            logger.info(f"Phase 1 complete: {len(audio_files)} videos ready ({phase1_time:.1f}s)")

        # =========================================================================
        # PHASE 2: Sequential GPU transcription (mutex protected)
        # =========================================================================
        if show_progress:
            logger.info("Phase 2: Transcribing with shared model (sequential GPU)...")

        phase2_start = time.time()
        total = len(audio_files)

        for i, (video_path, audio_path) in enumerate(audio_files.items()):
            video_name = Path(video_path).stem[:40]

            if show_progress and (i + 1) % progress_log_interval == 0 or i == 0 or i == total - 1:
                pct = ((i + 1) / total) * 100
                elapsed = time.time() - phase2_start
                eta = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else 0
                logger.info(f"[{i+1}/{total}] {pct:.0f}% - {video_name} - ETA: {eta:.0f}s")

            # Get audio duration for speed ratio calculation (US-60-009)
            audio_duration = get_audio_duration(audio_path) or 0.0
            transcription_start = time.time()

            try:
                # Transcribe with WhisperClient (GPU-locked)
                raw_segments = whisper_client.transcribe(
                    audio_path,
                    language=language,
                    vad_filter=vad_filter,
                    min_silence_duration_ms=min_silence_duration_ms,
                    speech_pad_ms=speech_pad_ms
                )

                transcription_time = time.time() - transcription_start

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

            except Exception as e:
                # Log full traceback for debugging transcription issues
                logger.exception(f"  Transcription error for {video_name}: {e}")
                results[video_path] = []
                metrics.record_failure(video_path)

            # Clean up audio file
            try:
                Path(audio_path).unlink()
            except (OSError, IOError) as e:
                # Non-critical: temp file cleanup failure won't affect results
                logger.debug(f"Could not remove temp audio file {audio_path}: {e}")

        phase2_time = time.time() - phase2_start
        if show_progress:
            logger.info(f"Phase 2 complete: {len(results)} videos ({phase2_time:.1f}s)")

        # Record phase times in metrics (US-60-009)
        metrics.set_phase_times(phase1_time, phase2_time)

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
    cpu_threads: int = 4
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
        model_name=model_name, compute_type=compute_type,
        gpu_transcription_timeout=gpu_transcription_timeout,
        num_workers=num_workers, cpu_threads=cpu_threads
    )
    raw_segments = None
    last_error = None

    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            raw_segments = whisper_client.transcribe(
                audio_path,
                language=language,
                vad_filter=vad_filter,
                min_silence_duration_ms=min_silence_duration_ms,
                speech_pad_ms=speech_pad_ms
            )
            break  # Success, exit retry loop

        except Exception as e:
            last_error = e

            # Check if error is transient (worth retrying)
            if not is_transient_error(e):
                # Permanent error - don't retry
                logger.error(f"  Transcription failed for {video_name} (permanent error): {e}")
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
    compute_type: str = "auto",
    language: str = None,
    vad_filter: bool = True,  # Enable VAD by default for voiceover - better gap detection
    gpu_transcription_timeout: int = 300,
    num_workers: int = 1,
    cpu_threads: int = 4
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
        model_name=model_name, compute_type=compute_type,
        gpu_transcription_timeout=gpu_transcription_timeout,
        num_workers=num_workers, cpu_threads=cpu_threads
    )
    logger.info(f"Transcribing voiceover with VAD={'enabled' if vad_filter else 'disabled'}")
    return whisper_client.transcribe(
        audio_path,
        language=language,
        vad_filter=vad_filter
    )


def transcribe_voiceover_media(
    media_path: str,
    output_srt_path: str = None,
    model_name: str = "base",
    language: str = None,
    compute_type: str = "auto",
    cache_dir: str = None,
    word_timestamps: bool = True,
    vad_filter: bool = True,  # Enable VAD by default for voiceover - better gap detection
    gpu_transcription_timeout: int = 300,
    audio_extraction_timeout: int = 60,
    num_workers: int = 1,
    cpu_threads: int = 4
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
        model_name=model_name, compute_type=compute_type,
        gpu_transcription_timeout=gpu_transcription_timeout,
        num_workers=num_workers, cpu_threads=cpu_threads
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
        segments = whisper_client.transcribe(
            audio_path,
            language=language,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps
        )

        # Clean up extracted audio
        try:
            Path(audio_path).unlink()
        except (OSError, IOError) as e:
            # Non-critical: temp file cleanup failure won't affect results
            logger.debug(f"Could not remove extracted audio file {audio_path}: {e}")

    elif media_path.suffix.lower() in audio_extensions:
        # It's already an audio file
        logger.info(f"Transcribing voiceover with VAD={'enabled' if vad_filter else 'disabled'}")
        segments = whisper_client.transcribe(
            str(media_path),
            language=language,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps
        )
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
    compute_type: str = "auto"
) -> List[TranscriptSegment]:
    """
    Get transcript segments for a single video.
    Backward-compatible function.
    """
    cache = TranscriptCache(cache_dir)
    return transcribe_video(video_path, cache, model_name, compute_type)
