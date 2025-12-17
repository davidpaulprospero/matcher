#!/usr/bin/env python3
"""
Optimized Transcription Module - v2.4.1 (GPU Lock Fix)

CRITICAL FIX (v2.4.1):
======================
The _lock error was caused by multiple ThreadPoolExecutor workers trying to
simultaneously access the GPU through separate WhisperModel instances.
ctranslate2 (used by faster-whisper) cannot handle concurrent GPU access.

SOLUTION:
- Use a SINGLE shared WhisperModel instance
- Protect GPU transcription with a threading.Lock() mutex
- Parallelize only the CPU-bound work (audio extraction, I/O)
- Serialize GPU transcription through the shared model

This maintains most of the performance benefit (parallel audio extraction)
while eliminating the GPU concurrency issue.

CHANGES FROM v2.4:
==================
1. SHARED MODEL: Single WhisperModel instance shared across all workers
2. GPU MUTEX: threading.Lock() protects all GPU transcription calls
3. PARALLEL I/O: Audio extraction still parallelized (CPU-bound)
4. SEQUENTIAL GPU: Transcription serialized through mutex

INTEGRATION:
- Replace `from src.transcription_optimized import transcribe_videos_parallel`
- With `from src.transcription_optimized import transcribe_videos_parallel`
"""

import os
import sys
import json
import hashlib
import logging
import time
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue

logger = logging.getLogger(__name__)

# =============================================================================
# CRITICAL FIX: Global shared model and GPU mutex
# =============================================================================
_gpu_lock = threading.Lock()  # Mutex for GPU access
_shared_model = None          # Shared WhisperModel instance
_model_config = {}            # Model configuration cache

# Thread-safe progress tracking
_progress_lock = threading.Lock()
_progress_data = {"completed": 0, "total": 0, "current": ""}


def _get_shared_model(model_name: str, compute_type: str):
    """
    Get or create the shared WhisperModel instance.
    Thread-safe initialization with double-checked locking.
    """
    global _shared_model, _model_config
    
    # Check if we need to (re)initialize
    current_config = {"model": model_name, "compute_type": compute_type}
    
    if _shared_model is not None and _model_config == current_config:
        return _shared_model
    
    with _gpu_lock:
        # Double-check after acquiring lock
        if _shared_model is not None and _model_config == current_config:
            return _shared_model
        
        logger.info(f"  Initializing shared WhisperModel ({model_name}, {compute_type})...")
        
        try:
            from faster_whisper import WhisperModel
            
            # Determine device
            device = "cuda" if compute_type != "int8" else "cpu"
            if compute_type == "auto":
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    if device == "cuda":
                        compute_type = "float16"
                    else:
                        compute_type = "int8"
                except ImportError:
                    device = "cpu"
                    compute_type = "int8"
            
            _shared_model = WhisperModel(
                model_name, 
                device=device,
                compute_type=compute_type,
                num_workers=1,  # Single worker to prevent internal threading issues
                cpu_threads=4
            )
            _model_config = current_config
            
            logger.info(f"  ✓ Model loaded on {device} ({compute_type})")
            return _shared_model
            
        except Exception as e:
            logger.error(f"  Failed to load WhisperModel: {e}")
            raise


def _cleanup_shared_model():
    """Release the shared model (call at end of batch processing)"""
    global _shared_model, _model_config
    with _gpu_lock:
        _shared_model = None
        _model_config = {}


@dataclass
class SRTSegment:
    """Represents a single SRT segment"""
    index: int
    start_time: float
    end_time: float
    text: str
    source_file: str = ""
    
    def to_dict(self):
        return asdict(self)


@dataclass
class VideoIndex:
    """Master index entry for a processed video"""
    file_hash: str
    file_path: str
    file_size: int
    file_mtime: float
    transcript_path: str
    segment_count: int
    processed_at: float
    duration_seconds: float = 0.0
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> 'VideoIndex':
        return cls(**d)


class DeltaAwareIndex:
    """
    Maintains a master index of all processed videos.
    
    KEY FEATURE: During confidence enforcement retries, we only process
    NEW videos. Existing transcripts are loaded instantly from cache.
    """
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.index_path = self.cache_dir / "video_master_index.json"
        self.transcripts_dir = self.cache_dir / "transcriptions"
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing index
        self.index: Dict[str, VideoIndex] = {}
        self._load_index()
    
    def _load_index(self):
        """Load the master index from disk"""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r') as f:
                    data = json.load(f)
                    self.index = {
                        k: VideoIndex.from_dict(v) 
                        for k, v in data.items()
                    }
                logger.info(f"Loaded master index: {len(self.index)} videos")
            except Exception as e:
                logger.warning(f"Could not load master index: {e}")
                self.index = {}
    
    def _save_index(self):
        """Save the master index to disk"""
        try:
            with open(self.index_path, 'w') as f:
                json.dump(
                    {k: v.to_dict() for k, v in self.index.items()},
                    f, indent=2
                )
        except Exception as e:
            logger.warning(f"Could not save master index: {e}")
    
    def get_file_hash(self, file_path: str) -> str:
        """Get fast hash for file identification (uses size + mtime + partial content)"""
        p = Path(file_path)
        stat = p.stat()
        
        # Fast hash: combine size, mtime, and first/last 64KB
        hash_data = f"{stat.st_size}:{stat.st_mtime}".encode()
        
        with open(file_path, 'rb') as f:
            hash_data += f.read(65536)  # First 64KB
            if stat.st_size > 131072:
                f.seek(-65536, 2)
                hash_data += f.read(65536)  # Last 64KB
        
        return hashlib.md5(hash_data).hexdigest()
    
    def is_cached(self, video_path: str) -> bool:
        """Check if video is already in the index with valid cache"""
        try:
            file_hash = self.get_file_hash(video_path)
            
            if file_hash not in self.index:
                return False
            
            entry = self.index[file_hash]
            transcript_path = Path(entry.transcript_path)
            
            # Verify transcript still exists
            if not transcript_path.exists():
                del self.index[file_hash]
                return False
            
            return True
        except Exception:
            return False
    
    def get_cached_transcript(self, video_path: str) -> Optional[List[SRTSegment]]:
        """Load transcript from cache if available"""
        try:
            file_hash = self.get_file_hash(video_path)
            
            if file_hash not in self.index:
                return None
            
            entry = self.index[file_hash]
            transcript_path = Path(entry.transcript_path)
            
            if not transcript_path.exists():
                return None
            
            with open(transcript_path, 'r') as f:
                cached = json.load(f)
                return [SRTSegment(**seg) for seg in cached]
        except Exception as e:
            logger.debug(f"Could not load cached transcript: {e}")
            return None
    
    def add_to_index(
        self, 
        video_path: str, 
        segments: List[SRTSegment],
        duration: float = 0.0
    ):
        """Add a video and its transcript to the index"""
        try:
            file_hash = self.get_file_hash(video_path)
            p = Path(video_path)
            stat = p.stat()
            
            # Save transcript
            transcript_path = self.transcripts_dir / f"{file_hash}.json"
            with open(transcript_path, 'w') as f:
                json.dump([seg.to_dict() for seg in segments], f)
            
            # Update index
            self.index[file_hash] = VideoIndex(
                file_hash=file_hash,
                file_path=str(video_path),
                file_size=stat.st_size,
                file_mtime=stat.st_mtime,
                transcript_path=str(transcript_path),
                segment_count=len(segments),
                processed_at=time.time(),
                duration_seconds=duration
            )
            
            # Save index periodically (every 10 videos)
            if len(self.index) % 10 == 0:
                self._save_index()
                
        except Exception as e:
            logger.warning(f"Could not add to index: {e}")
    
    def finalize(self):
        """Save the index to disk"""
        self._save_index()
    
    def get_new_videos(self, video_paths: List[str]) -> Tuple[List[str], List[str]]:
        """
        Separate videos into new (need processing) and cached.
        
        Returns: (new_videos, cached_videos)
        """
        new_videos = []
        cached_videos = []
        
        for path in video_paths:
            if self.is_cached(path):
                cached_videos.append(path)
            else:
                new_videos.append(path)
        
        return new_videos, cached_videos
    
    def get_stats(self) -> dict:
        """Get index statistics"""
        return {
            "total_videos": len(self.index),
            "total_segments": sum(e.segment_count for e in self.index.values()),
            "total_duration_hours": sum(e.duration_seconds for e in self.index.values()) / 3600
        }


def _get_video_duration(video_path: str) -> float:
    """Get video duration using ffprobe"""
    import subprocess
    
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def _extract_audio_safe(video_path: str, cache_dir: str) -> Optional[str]:
    """
    Extract audio from video with error tolerance.
    This is CPU-bound and can be parallelized safely.
    """
    import subprocess
    
    # Create audio cache directory
    audio_dir = Path(cache_dir) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Use video filename for audio cache (faster than hashing)
    video_name = Path(video_path).stem
    audio_path = audio_dir / f"{video_name}.wav"
    
    if audio_path.exists() and audio_path.stat().st_size > 1000:
        return str(audio_path)
    
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-err_detect", "ignore_err",
        "-fflags", "+genpts+igndts",
        str(audio_path)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if audio_path.exists() and audio_path.stat().st_size > 1000:
            return str(audio_path)
        return None
        
    except Exception as e:
        logger.debug(f"Audio extraction failed: {e}")
        return None


def _transcribe_audio_with_shared_model(
    audio_path: str,
    video_path: str,
    model_name: str,
    language: str,
    compute_type: str
) -> List[SRTSegment]:
    """
    Transcribe audio using the shared WhisperModel with GPU mutex.
    
    CRITICAL: This function acquires the GPU lock to prevent concurrent access.
    """
    global _gpu_lock
    
    # Acquire GPU lock for transcription
    with _gpu_lock:
        try:
            model = _get_shared_model(model_name, compute_type)
            
            # Transcribe (GPU-bound operation)
            segments_iter, info = model.transcribe(
                audio_path,
                language=language,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=400
                )
            )
            
            # Convert to SRTSegment (must consume iterator while holding lock)
            segments = []
            for i, seg in enumerate(segments_iter, 1):
                text = seg.text.strip()
                if text:
                    segments.append(SRTSegment(
                        index=i,
                        start_time=seg.start,
                        end_time=seg.end,
                        text=text,
                        source_file=video_path
                    ))
            
            return segments
            
        except Exception as e:
            logger.error(f"Transcription error for {Path(video_path).name}: {e}")
            return []


def _transcribe_with_whisper_fallback(
    audio_path: str,
    video_path: str,
    model_name: str
) -> List[SRTSegment]:
    """Fallback to OpenAI whisper if faster-whisper fails"""
    global _gpu_lock
    
    with _gpu_lock:
        try:
            import whisper
            
            model = whisper.load_model(model_name)
            result = model.transcribe(audio_path, verbose=False)
            
            segments = []
            for i, seg in enumerate(result['segments'], 1):
                text = seg['text'].strip()
                if text:
                    segments.append(SRTSegment(
                        index=i,
                        start_time=seg['start'],
                        end_time=seg['end'],
                        text=text,
                        source_file=video_path
                    ))
            
            return segments
            
        except Exception as e:
            logger.error(f"Whisper fallback failed: {e}")
            return []


def _prepare_video_for_transcription(
    video_path: str,
    cache_dir: str
) -> Tuple[str, Optional[str], float]:
    """
    Prepare a video for transcription (extract audio, get duration).
    This is CPU-bound and can be safely parallelized.
    
    Returns: (video_path, audio_path, duration)
    """
    video_name = Path(video_path).name
    
    try:
        # Get video duration
        duration = _get_video_duration(video_path)
        
        # Extract audio (CPU-bound, parallel-safe)
        audio_path = _extract_audio_safe(video_path, cache_dir)
        
        if audio_path is None:
            logger.warning(f"  Could not extract audio: {video_name}")
            return video_path, None, duration
        
        return video_path, audio_path, duration
        
    except Exception as e:
        logger.error(f"  Preparation failed for {video_name}: {e}")
        return video_path, None, 0.0


def transcribe_videos_parallel(
    video_paths: List[str],
    cache: Any,  # CacheManager from utils
    config: Any,
    max_workers: int = 4,
    force_reprocess: bool = False
) -> Dict[str, List[SRTSegment]]:
    """
    Transcribe videos with parallel audio extraction and serialized GPU transcription.
    
    ARCHITECTURE (v2.4.1 - GPU Lock Fix):
    =====================================
    1. Phase 1 (Parallel): Extract audio from all videos using ThreadPoolExecutor
    2. Phase 2 (Sequential): Transcribe audio using shared model with GPU mutex
    
    This approach:
    - Parallelizes CPU-bound audio extraction (significant speedup)
    - Serializes GPU transcription (prevents _lock errors)
    - Maintains delta-aware caching (skip processed videos)
    
    Args:
        video_paths: List of video file paths
        cache: CacheManager instance
        config: Configuration object
        max_workers: Number of parallel workers for audio extraction
        force_reprocess: If True, ignore cache and reprocess all
    
    Returns:
        Dict mapping video_path -> List[SRTSegment]
    """
    # VERSION CHECK - This helps verify correct file is loaded
    import sys
    print("  ╔══════════════════════════════════════════════════════════════╗", flush=True)
    print("  ║  transcription_optimized v2.4.1-debug (GPU Lock Fix)        ║", flush=True)
    print("  ╚══════════════════════════════════════════════════════════════╝", flush=True)
    sys.stdout.flush()
    
    global _progress_data
    
    # Get config values
    cache_dir = config.cache_dir if hasattr(config, 'cache_dir') else str(cache.cache_dir)
    model_name = getattr(config.transcription, 'model', 'base')
    language = getattr(config.transcription, 'language', 'en')
    compute_type = getattr(config.transcription, 'compute_type', 'auto')
    
    # Initialize delta-aware index
    delta_index = DeltaAwareIndex(cache_dir)
    
    # Separate new vs cached videos
    if force_reprocess:
        new_videos = video_paths
        cached_videos = []
    else:
        new_videos, cached_videos = delta_index.get_new_videos(video_paths)
    
    logger.info(f"Video index: {len(cached_videos)} cached, {len(new_videos)} new")
    
    # Load cached transcripts
    all_transcripts = {}
    for video_path in cached_videos:
        segments = delta_index.get_cached_transcript(video_path)
        if segments:
            all_transcripts[video_path] = segments
    
    if cached_videos:
        logger.info(f"  ✓ Loaded {len(all_transcripts)} cached transcripts instantly")
    
    # Process new videos
    if not new_videos:
        return all_transcripts
    
    logger.info(f"  Processing {len(new_videos)} new videos...")
    start_time = time.time()
    
    # =========================================================================
    # PHASE 1: Parallel audio extraction (CPU-bound, safe to parallelize)
    # =========================================================================
    logger.info(f"  Phase 1: Extracting audio ({max_workers} workers)...")
    phase1_start = time.time()
    
    prepared_videos = []  # List of (video_path, audio_path, duration)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _prepare_video_for_transcription,
                video_path,
                cache_dir
            ): video_path
            for video_path in new_videos
        }
        
        for future in as_completed(futures):
            video_path = futures[future]
            try:
                result = future.result()
                if result[1] is not None:  # audio_path is not None
                    prepared_videos.append(result)
                else:
                    logger.warning(f"  Skipping {Path(video_path).name} (no audio)")
            except Exception as e:
                logger.error(f"  Preparation failed for {Path(video_path).name}: {e}")
    
    phase1_time = time.time() - phase1_start
    logger.info(f"  ✓ Phase 1 complete: {len(prepared_videos)} videos ready ({phase1_time:.1f}s)")
    
    # =========================================================================
    # PHASE 2: Sequential transcription (GPU-bound, must be serialized)
    # =========================================================================
    logger.info(f"  Phase 2: Transcribing with shared model (sequential GPU)...")
    phase2_start = time.time()
    
    # Initialize shared model before starting
    try:
        _get_shared_model(model_name, compute_type)
    except Exception as e:
        logger.error(f"  Failed to initialize model: {e}")
        logger.info("  Falling back to OpenAI Whisper...")
    
    # Progress tracking
    _progress_data = {"completed": 0, "total": len(prepared_videos), "current": ""}
    
    for video_path, audio_path, duration in prepared_videos:
        video_name = Path(video_path).name
        
        with _progress_lock:
            _progress_data["current"] = video_name
        
        try:
            # Transcribe with shared model (GPU mutex inside)
            segments = _transcribe_audio_with_shared_model(
                audio_path,
                video_path,
                model_name,
                language,
                compute_type
            )
            
            if segments:
                all_transcripts[video_path] = segments
                delta_index.add_to_index(video_path, segments, duration)
            
            # Update progress
            with _progress_lock:
                _progress_data["completed"] += 1
                pct = (_progress_data["completed"] / _progress_data["total"]) * 100
                elapsed = time.time() - phase2_start
                if _progress_data["completed"] > 0:
                    eta = (elapsed / _progress_data["completed"]) * (_progress_data["total"] - _progress_data["completed"])
                else:
                    eta = 0
                logger.info(f"  [{_progress_data['completed']}/{_progress_data['total']}] {pct:.0f}% - {video_name} ({len(segments)} segments) - ETA: {eta:.0f}s")
        
        except Exception as e:
            logger.error(f"  Transcription failed for {video_name}: {e}")
            with _progress_lock:
                _progress_data["completed"] += 1
    
    phase2_time = time.time() - phase2_start
    total_time = time.time() - start_time
    
    logger.info(f"  ✓ Phase 2 complete: {_progress_data['completed']} videos ({phase2_time:.1f}s)")
    logger.info(f"  ✓ Total processing time: {total_time:.1f}s")
    
    # AGGRESSIVE DEBUG: Use print with flush to catch silent crashes
    import sys
    print("  [DEBUG] Starting cleanup...", flush=True)
    sys.stdout.flush()
    
    # Finalize index with error handling
    try:
        print("  [DEBUG] Saving index to disk...", flush=True)
        sys.stdout.flush()
        delta_index.finalize()
        print("  [DEBUG] ✓ Index saved", flush=True)
        sys.stdout.flush()
    except Exception as e:
        print(f"  [DEBUG] ✗ Failed to save index: {e}", flush=True)
        sys.stdout.flush()
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
    
    # Cleanup shared model to free GPU memory
    # NOTE: DISABLED - ctranslate2 segfaults during model cleanup on some systems
    # The model will be garbage collected when the process ends anyway
    try:
        print("  [DEBUG] Skipping GPU model release (causes segfault on some systems)", flush=True)
        sys.stdout.flush()
        # _cleanup_shared_model()  # DISABLED - causes crash
        print("  [DEBUG] ✓ GPU cleanup skipped (will be GC'd at process end)", flush=True)
        sys.stdout.flush()
    except Exception as e:
        print(f"  [DEBUG] ✗ Failed to release model: {e}", flush=True)
        sys.stdout.flush()
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
    
    # Log stats
    try:
        print("  [DEBUG] Getting stats...", flush=True)
        sys.stdout.flush()
        stats = delta_index.get_stats()
        print(f"  [DEBUG] Index stats: {stats['total_videos']} videos, {stats['total_segments']} segments", flush=True)
        sys.stdout.flush()
    except Exception as e:
        print(f"  [DEBUG] ✗ Failed to get stats: {e}", flush=True)
        sys.stdout.flush()
    
    print(f"  [DEBUG] Returning {len(all_transcripts)} transcripts to main pipeline...", flush=True)
    sys.stdout.flush()
    
    return all_transcripts


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

def transcribe_voiceover_audio(
    audio_path: str,
    output_srt_path: str,
    model_name: str = "base",
    language: str = "en",
    compute_type: str = "auto"
) -> str:
    """
    Alias for transcribe_voiceover_media for backward compatibility.
    Some modules import this name instead of transcribe_voiceover_media.
    """
    return transcribe_voiceover_media(
        media_path=audio_path,
        output_srt_path=output_srt_path,
        model_name=model_name,
        language=language,
        compute_type=compute_type
    )


def transcribe_all_videos(config, force_reprocess: bool = False) -> Dict[str, List[SRTSegment]]:
    """
    Legacy wrapper for transcribe_videos_parallel.
    Scans video directory and transcribes all videos.
    """
    from src.utils import CacheManager
    
    video_dir = Path(config.downloaded_videos_dir)
    video_extensions = {'.mp4', '.mkv', '.webm', '.avi', '.mov', '.mxf'}
    
    video_paths = []
    for ext in video_extensions:
        video_paths.extend([str(p) for p in video_dir.rglob(f'*{ext}')])
    
    logger.info(f"Found {len(video_paths)} videos")
    
    cache = CacheManager(config.cache_dir)
    return transcribe_videos_parallel(
        video_paths, cache, config, 
        max_workers=4,
        force_reprocess=force_reprocess
    )


def transcribe_voiceover_media(
    media_path: str,
    output_srt_path: str,
    model_name: str = "base",
    language: str = "en",
    compute_type: str = "auto"
) -> str:
    """Transcribe voiceover audio/video to SRT"""
    cache_dir = str(Path(media_path).parent / ".cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract audio
    audio_path = _extract_audio_safe(media_path, cache_dir)
    if audio_path is None:
        audio_path = media_path
    
    # Transcribe with shared model
    segments = _transcribe_audio_with_shared_model(
        audio_path, media_path, model_name, language, compute_type
    )
    
    # Cleanup model after single-file transcription
    # DISABLED - causes segfault on some systems
    # _cleanup_shared_model()
    
    # Write SRT
    with open(output_srt_path, 'w', encoding='utf-8') as f:
        for seg in segments:
            f.write(f"{seg.index}\n")
            f.write(f"{_format_timestamp(seg.start_time)} --> {_format_timestamp(seg.end_time)}\n")
            f.write(f"{seg.text}\n\n")
    
    return output_srt_path


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')


# =============================================================================
# ALTERNATIVE: Process Queue Approach (for very large batches)
# =============================================================================

class TranscriptionQueue:
    """
    Alternative approach using a queue for very large batches.
    Use this if you need more control over GPU scheduling.
    """
    
    def __init__(self, model_name: str, compute_type: str, language: str):
        self.model_name = model_name
        self.compute_type = compute_type
        self.language = language
        self.queue = queue.Queue()
        self.results = {}
        self.lock = threading.Lock()
        self._running = False
        self._worker_thread = None
    
    def start(self):
        """Start the transcription worker thread"""
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
    
    def stop(self):
        """Stop the worker thread"""
        self._running = False
        self.queue.put(None)  # Poison pill
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
    
    def _worker(self):
        """Worker thread that processes transcription requests sequentially"""
        model = _get_shared_model(self.model_name, self.compute_type)
        
        while self._running:
            try:
                item = self.queue.get(timeout=1)
                if item is None:
                    break
                
                video_path, audio_path = item
                
                try:
                    segments_iter, info = model.transcribe(
                        audio_path,
                        language=self.language,
                        vad_filter=True
                    )
                    
                    segments = []
                    for i, seg in enumerate(segments_iter, 1):
                        text = seg.text.strip()
                        if text:
                            segments.append(SRTSegment(
                                index=i,
                                start_time=seg.start,
                                end_time=seg.end,
                                text=text,
                                source_file=video_path
                            ))
                    
                    with self.lock:
                        self.results[video_path] = segments
                        
                except Exception as e:
                    logger.error(f"Queue transcription failed: {e}")
                    with self.lock:
                        self.results[video_path] = []
                
                self.queue.task_done()
                
            except queue.Empty:
                continue
    
    def submit(self, video_path: str, audio_path: str):
        """Submit a video for transcription"""
        self.queue.put((video_path, audio_path))
    
    def get_result(self, video_path: str) -> Optional[List[SRTSegment]]:
        """Get transcription result for a video"""
        with self.lock:
            return self.results.get(video_path)
    
    def wait_all(self):
        """Wait for all queued items to be processed"""
        self.queue.join()