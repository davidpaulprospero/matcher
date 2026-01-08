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
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from .whisper_client import WhisperClient
from .cache import TranscriptCache
from .utils import extract_audio, write_srt
from src.state import TranscriptSegment

logger = logging.getLogger(__name__)


def transcribe_videos_parallel(
    video_paths: List[str],
    cache: Any,
    config: Any = None,
    max_workers: int = 4,
    force_reprocess: bool = False,
    show_progress: bool = True
) -> Dict[str, List[TranscriptSegment]]:
    """
    Transcribe multiple videos with parallel audio extraction but sequential GPU.

    TWO-PHASE PROCESSING:
    Phase 1: Parallel audio extraction (CPU-bound, safe to parallelize)
    Phase 2: Sequential GPU transcription (must be serialized)

    Args:
        video_paths: List of video file paths
        cache: CacheManager or similar with cache_dir attribute
        config: Configuration object with transcription settings
        max_workers: Number of parallel workers for audio extraction
        force_reprocess: If True, ignore cache and reprocess all
        show_progress: Whether to show progress

    Returns:
        Dict mapping video path to list of TranscriptSegments
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
        vad_filter = getattr(config.transcription, 'vad_filter', True)
        min_silence_duration_ms = getattr(config.transcription, 'min_silence_duration_ms', 200)
        speech_pad_ms = getattr(config.transcription, 'speech_pad_ms', 10)
    else:
        model_name = "base"
        compute_type = "auto"
        language = None
        vad_filter = True
        min_silence_duration_ms = 200
        speech_pad_ms = 10

    # Initialize WhisperClient and TranscriptCache
    whisper_client = WhisperClient(model_name=model_name, compute_type=compute_type)
    transcript_cache = TranscriptCache(cache_dir)
    results = {}

    # Separate cached vs uncached
    cached_videos = []
    uncached_videos = []

    for video_path in video_paths:
        if force_reprocess:
            uncached_videos.append(video_path)
        else:
            cached = transcript_cache.get(video_path)
            if cached:
                cached_videos.append((video_path, cached))
            else:
                uncached_videos.append(video_path)

    # Load cached results
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

    if show_progress:
        print(f"  Video index: {len(cached_videos)} cached, {len(uncached_videos)} new", flush=True)

    if not uncached_videos:
        return results

    # Create temp directory for audio files
    temp_dir = Path(cache_dir) / "temp_audio"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # PHASE 1: Parallel audio extraction (CPU-bound)
    # =========================================================================
    if show_progress:
        print(f"  Phase 1: Extracting audio ({max_workers} workers)...", flush=True)

    audio_files = {}  # video_path -> audio_path
    phase1_start = time.time()

    def extract_audio_task(video_path):
        audio_path = extract_audio(video_path, str(temp_dir))
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
                if show_progress and completed % 10 == 0:
                    print(f"    Extracted {completed}/{total_videos} audio files...", flush=True)
            except Exception as e:
                completed += 1
                logger.error(f"  Audio extraction error: {e}")

    phase1_time = time.time() - phase1_start
    if show_progress:
        print(f"  ✓ Phase 1 complete: {len(audio_files)} videos ready ({phase1_time:.1f}s)", flush=True)

    # =========================================================================
    # PHASE 2: Sequential GPU transcription (mutex protected)
    # =========================================================================
    if show_progress:
        print(f"  Phase 2: Transcribing with shared model (sequential GPU)...", flush=True)

    phase2_start = time.time()
    total = len(audio_files)

    for i, (video_path, audio_path) in enumerate(audio_files.items()):
        video_name = Path(video_path).stem[:40]

        if show_progress:
            pct = ((i + 1) / total) * 100
            elapsed = time.time() - phase2_start
            eta = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else 0
            print(f"\r  [{i+1}/{total}] {pct:.0f}% - {video_name} - ETA: {eta:.0f}s    ", end='', flush=True)

        try:
            # Transcribe with WhisperClient (GPU-locked)
            raw_segments = whisper_client.transcribe(
                audio_path,
                language=language,
                vad_filter=vad_filter,
                min_silence_duration_ms=min_silence_duration_ms,
                speech_pad_ms=speech_pad_ms
            )

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

        except Exception as e:
            logger.error(f"  Transcription error for {video_name}: {e}")
            results[video_path] = []

        # Clean up audio file
        try:
            Path(audio_path).unlink()
        except:
            pass

    if show_progress:
        print()  # New line after progress

    phase2_time = time.time() - phase2_start
    if show_progress:
        print(f"  ✓ Phase 2 complete: {len(results)} videos ({phase2_time:.1f}s)", flush=True)

    # Clean up temp directory
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except:
        pass

    return results


def transcribe_video(
    video_path: str,
    cache: TranscriptCache,
    model_name: str = "base",
    compute_type: str = "auto",
    language: str = None,
    temp_dir: str = None,
    vad_filter: bool = True,
    min_silence_duration_ms: int = 200,
    speech_pad_ms: int = 10
) -> List[TranscriptSegment]:
    """
    Transcribe a single video file.
    Uses cache if available, otherwise transcribes with shared model.

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
    """
    video_path = str(video_path)
    video_name = Path(video_path).name

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
    audio_path = extract_audio(video_path, temp_dir)
    if not audio_path:
        logger.warning(f"  Could not extract audio: {video_name}")
        return []

    # Transcribe with WhisperClient (GPU-locked)
    try:
        whisper_client = WhisperClient(model_name=model_name, compute_type=compute_type)
        raw_segments = whisper_client.transcribe(
            audio_path,
            language=language,
            vad_filter=vad_filter,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms
        )

        # Cache the result
        cache.set(video_path, raw_segments)

        # Clean up audio file
        try:
            Path(audio_path).unlink()
        except:
            pass

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

    except Exception as e:
        logger.error(f"  Transcription failed for {video_name}: {e}")
        return []


def transcribe_voiceover_audio(
    audio_path: str,
    model_name: str = "base",
    compute_type: str = "auto",
    language: str = None
) -> List[dict]:
    """
    Transcribe a voiceover audio file.
    Backward-compatible function for voiceover transcription.
    """
    whisper_client = WhisperClient(model_name=model_name, compute_type=compute_type)
    return whisper_client.transcribe(
        audio_path,
        language=language,
        vad_filter=False  # Don't filter voiceover
    )


def transcribe_voiceover_media(
    media_path: str,
    output_srt_path: str = None,
    model_name: str = "base",
    language: str = None,
    compute_type: str = "auto",
    cache_dir: str = None,
    word_timestamps: bool = True
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
    whisper_client = WhisperClient(model_name=model_name, compute_type=compute_type)

    if media_path.suffix.lower() in video_extensions:
        # Extract audio first
        if cache_dir:
            temp_dir = Path(cache_dir) / "temp_audio"
            temp_dir.mkdir(parents=True, exist_ok=True)
        else:
            temp_dir = media_path.parent

        audio_path = extract_audio(str(media_path), str(temp_dir))

        if not audio_path:
            logger.error(f"Could not extract audio from {media_path}")
            raise RuntimeError(f"Could not extract audio from {media_path}")

        # Transcribe the extracted audio with word timestamps
        segments = whisper_client.transcribe(
            audio_path,
            language=language,
            vad_filter=False,  # Don't filter voiceover
            word_timestamps=word_timestamps
        )

        # Clean up extracted audio
        try:
            Path(audio_path).unlink()
        except:
            pass

    elif media_path.suffix.lower() in audio_extensions:
        # It's already an audio file
        segments = whisper_client.transcribe(
            str(media_path),
            language=language,
            vad_filter=False,
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
