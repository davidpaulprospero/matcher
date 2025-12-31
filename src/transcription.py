#!/usr/bin/env python3
"""
Transcription Module - v2.5 (GPU Lock Fix)

CRITICAL FIX:
=============
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
_gpu_lock = threading.RLock()  # RLock is REENTRANT - allows same thread to acquire multiple times
_shared_model = None          # Shared WhisperModel instance
_model_config = {}            # Model configuration cache

# Thread-safe progress tracking
_progress_lock = threading.Lock()
_progress_data = {"completed": 0, "total": 0, "current": ""}


class DeltaAwareIndex:
    """
    Tracks which videos have been indexed to enable delta-aware processing.
    Only processes NEW videos on subsequent runs.
    """
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.index_path = self.cache_dir / "delta_index.json"
        self.indexed_videos = set()
        self._load()
    
    def _load(self):
        """Load the index of previously processed videos"""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r') as f:
                    data = json.load(f)
                    self.indexed_videos = set(data.get('indexed', []))
            except Exception as e:
                logger.debug(f"Could not load delta index: {e}")
                self.indexed_videos = set()
    
    def _save(self):
        """Save the index"""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self.index_path, 'w') as f:
                json.dump({
                    'indexed': list(self.indexed_videos),
                    'updated_at': time.time()
                }, f)
        except Exception as e:
            logger.debug(f"Could not save delta index: {e}")
    
    def is_indexed(self, video_path: str) -> bool:
        """Check if a video has been indexed"""
        return str(video_path) in self.indexed_videos
    
    def mark_indexed(self, video_path: str):
        """Mark a video as indexed"""
        self.indexed_videos.add(str(video_path))
        self._save()
    
    def mark_indexed_batch(self, video_paths: List[str]):
        """Mark multiple videos as indexed"""
        for vp in video_paths:
            self.indexed_videos.add(str(vp))
        self._save()
    
    def get_new_videos(self, video_paths: List[str]) -> List[str]:
        """Get list of videos that haven't been indexed yet"""
        return [vp for vp in video_paths if not self.is_indexed(vp)]
    
    def clear(self):
        """Clear the index (force full reprocess)"""
        self.indexed_videos = set()
        self._save()


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
        
        print(f"\n  [MODEL] Initializing WhisperModel...", flush=True)
        print(f"  [MODEL]   Model: {model_name}", flush=True)
        print(f"  [MODEL]   Compute type: {compute_type}", flush=True)
        
        try:
            print(f"  [MODEL]   Step 1: Importing faster_whisper...", flush=True)
            from faster_whisper import WhisperModel
            print(f"  [MODEL]   Step 1: Done", flush=True)
            
            # Determine device
            device = "cuda"
            actual_compute = compute_type
            
            if compute_type == "auto" or compute_type == "int8":
                print(f"  [MODEL]   Step 2: Checking CUDA availability...", flush=True)
                try:
                    import torch
                    cuda_available = torch.cuda.is_available()
                    print(f"  [MODEL]   Step 2: CUDA available = {cuda_available}", flush=True)
                    if cuda_available:
                        device = "cuda"
                        actual_compute = "float16"
                        print(f"  [MODEL]   Step 2: GPU memory = {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB", flush=True)
                    else:
                        device = "cpu"
                        actual_compute = "int8"
                except ImportError:
                    print(f"  [MODEL]   Step 2: torch not available, using CPU", flush=True)
                    device = "cpu"
                    actual_compute = "int8"
            
            print(f"  [MODEL]   Step 3: Creating WhisperModel on {device} ({actual_compute})...", flush=True)
            print(f"  [MODEL]   (This may take 30-60 seconds on first run to download model)", flush=True)
            
            import sys
            sys.stdout.flush()
            sys.stderr.flush()
            
            _shared_model = WhisperModel(
                model_name, 
                device=device,
                compute_type=actual_compute,
                num_workers=1,
                cpu_threads=4
            )
            _model_config = current_config
            
            print(f"  [MODEL]   Step 3: Done!", flush=True)
            print(f"  [MODEL] ✓ Model ready on {device} ({actual_compute})\n", flush=True)
            
            return _shared_model
            
        except Exception as e:
            print(f"  [MODEL] ✗ Failed to load model: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise


def _transcribe_with_shared_model(
    audio_path: str,
    model_name: str,
    compute_type: str,
    language: str = None,
    vad_filter: bool = True,
    min_silence_duration_ms: int = 200,
    speech_pad_ms: int = 10
) -> List[dict]:
    """
    Transcribe audio using the shared model with mutex protection.

    Args:
        audio_path: Path to audio file
        model_name: Whisper model name (base, small, medium, large, etc.)
        compute_type: Compute type (auto, float16, int8)
        language: Language code or None for auto-detect
        vad_filter: Whether to apply Voice Activity Detection
        min_silence_duration_ms: Minimum silence duration to split segments (from config)
        speech_pad_ms: Padding around detected speech (from config)
    """
    print(f"\n  [TRANSCRIBE] Acquiring GPU lock...", flush=True)
    with _gpu_lock:
        print(f"  [TRANSCRIBE] Lock acquired, getting model...", flush=True)
        model = _get_shared_model(model_name, compute_type)
        
        audio_name = Path(audio_path).name[:40]
        print(f"  [TRANSCRIBE] Starting transcription of {audio_name}...", flush=True)
        try:
            segments, info = model.transcribe(
                audio_path,
                language=language,
                vad_filter=vad_filter,
                vad_parameters=dict(
                    min_silence_duration_ms=min_silence_duration_ms,
                    speech_pad_ms=speech_pad_ms
                )
            )
            
            print(f"  [TRANSCRIBE] Transcription done, processing segments...", flush=True)
            result = []
            for seg in segments:
                result.append({
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip()
                })
            
            print(f"  [TRANSCRIBE] Done: {len(result)} segments", flush=True)
            return result
            
        except Exception as e:
            print(f"  [TRANSCRIBE] Error: {e}", flush=True)
            logger.error(f"Transcription error: {e}")
            import traceback
            traceback.print_exc()
            return []


@dataclass
class TranscriptSegment:
    """A single transcript segment"""
    index: int
    start_time: float
    end_time: float
    text: str
    source_file: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TranscriptCache:
    """Cache for video transcripts"""
    cache_dir: Path
    alt_cache_dir: Path = None
    _source_map: Dict[str, Path] = None  # Maps source_file -> cache_file
    
    def __init__(self, cache_dir: str):
        base_dir = Path(cache_dir)
        
        # Check both possible folder names
        self.cache_dir = base_dir / "transcriptions"
        self.alt_cache_dir = base_dir / "transcripts"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._source_map = {}
        
        print(f"  [DEBUG] Primary cache: {self.cache_dir}", flush=True)
        print(f"  [DEBUG] Alt cache: {self.alt_cache_dir}", flush=True)
        
        # Build reverse lookup by reading source_file from each cache file
        self._build_source_map()
    
    def _build_source_map(self):
        """Build mapping from video path to cache file by reading source_file from each"""
        for folder in [self.cache_dir, self.alt_cache_dir]:
            if not folder or not folder.exists():
                continue
            
            cache_files = list(folder.glob("*.json"))
            print(f"  [DEBUG] Scanning {len(cache_files)} cache files in {folder.name}...", flush=True)
            
            for cache_file in cache_files:
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Extract source_file from the data
                    source_file = None
                    
                    if isinstance(data, list) and len(data) > 0:
                        # List of segments - get source_file from first segment
                        source_file = data[0].get('source_file', '')
                    elif isinstance(data, dict):
                        # Dict format - check various keys
                        source_file = data.get('source_file', data.get('video', data.get('video_path', '')))
                        if not source_file and 'segments' in data:
                            segs = data['segments']
                            if segs and len(segs) > 0:
                                source_file = segs[0].get('source_file', '')
                    
                    if source_file:
                        # Normalize path for matching
                        source_file = str(Path(source_file).resolve()) if source_file else ''
                        self._source_map[source_file] = cache_file
                        
                        # Also add just the filename as key for partial matching
                        filename = Path(source_file).name
                        self._source_map[filename] = cache_file
                        
                except Exception as e:
                    continue
            
        print(f"  [DEBUG] Built source map with {len(self._source_map)} entries", flush=True)
        if self._source_map:
            sample_keys = list(self._source_map.keys())[:2]
            for k in sample_keys:
                print(f"  [DEBUG]   Sample: {Path(k).name[:50]}... -> {self._source_map[k].name}", flush=True)
    
    def _get_video_hash(self, video_path: str) -> str:
        """Get hash for a video file based on path and size"""
        path = Path(video_path)
        size = path.stat().st_size if path.exists() else 0
        key = f"{path.name}:{size}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, video_path: str) -> Optional[List[dict]]:
        """Get cached transcript for a video"""
        video_path_resolved = str(Path(video_path).resolve())
        video_name = Path(video_path).name
        
        # Try source map lookup first (most reliable)
        cache_file = None
        if video_path_resolved in self._source_map:
            cache_file = self._source_map[video_path_resolved]
        elif video_name in self._source_map:
            cache_file = self._source_map[video_name]
        elif video_path in self._source_map:
            cache_file = self._source_map[video_path]
        
        # Fallback to hash-based lookup
        if not cache_file:
            video_hash = self._get_video_hash(video_path)
            for folder in [self.cache_dir, self.alt_cache_dir]:
                if folder and folder.exists():
                    exact = folder / f"{video_hash}.json"
                    if exact.exists():
                        cache_file = exact
                        break
        
        if not cache_file or not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different cache formats
            if isinstance(data, list):
                segments = data
            elif isinstance(data, dict):
                segments = data.get('segments', data.get('transcripts', []))
            else:
                return None
            
            # Normalize segment format
            normalized = []
            for seg in segments:
                if isinstance(seg, dict):
                    normalized.append({
                        'start': seg.get('start', seg.get('start_time', 0)),
                        'end': seg.get('end', seg.get('end_time', 0)),
                        'text': seg.get('text', '')
                    })
            
            return normalized if normalized else None
            
        except Exception as e:
            logger.debug(f"Cache read error: {e}")
            return None
    
    def set(self, video_path: str, segments: List[dict]):
        """Cache transcript for a video"""
        video_hash = self._get_video_hash(video_path)
        cache_file = self.cache_dir / f"{video_hash}.json"
        
        # Convert to format matching existing cache
        cache_data = []
        for i, seg in enumerate(segments):
            cache_data.append({
                'index': i + 1,
                'start_time': seg.get('start', seg.get('start_time', 0)),
                'end_time': seg.get('end', seg.get('end_time', 0)),
                'text': seg.get('text', ''),
                'source_file': str(video_path)
            })
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
            
            # Update source map
            video_path_resolved = str(Path(video_path).resolve())
            self._source_map[video_path_resolved] = cache_file
            self._source_map[Path(video_path).name] = cache_file
            
        except Exception as e:
            logger.debug(f"Could not cache transcript: {e}")


def extract_audio(video_path: str, output_dir: str = None) -> Optional[str]:
    """Extract audio from video file"""
    import subprocess
    import hashlib
    
    video_path = Path(video_path)
    
    # Use hash of full path to avoid collisions with similar filenames
    path_hash = hashlib.md5(str(video_path).encode()).hexdigest()[:8]
    audio_filename = f"{video_path.stem[:80]}_{path_hash}.wav"
    
    if output_dir:
        audio_path = Path(output_dir) / audio_filename
    else:
        audio_path = video_path.parent / audio_filename
    
    # Skip if already extracted
    if audio_path.exists():
        return str(audio_path)
    
    video_path = Path(video_path)
    if output_dir:
        audio_path = Path(output_dir) / f"{video_path.stem}.wav"
    else:
        audio_path = video_path.with_suffix('.wav')
    
    # Skip if already extracted
    if audio_path.exists():
        return str(audio_path)
    
    try:
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            '-y', str(audio_path)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0 and audio_path.exists():
            return str(audio_path)
        else:
            logger.debug(f"FFmpeg error: {result.stderr}")
            return None
            
    except Exception as e:
        logger.debug(f"Audio extraction error: {e}")
        return None


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
    
    # Transcribe with shared model (mutex protected)
    try:
        raw_segments = _transcribe_with_shared_model(
            audio_path,
            model_name,
            compute_type,
            language,
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
            print(f"\r  [{i+1}/{total}] {pct:.0f}% - {video_name} - ETA: {eta:.0f}s    ", flush=True)  # Added newline
        
        try:
            print(f"    >> Calling _transcribe_with_shared_model...", flush=True)
            raw_segments = _transcribe_with_shared_model(
                audio_path,
                model_name,
                compute_type,
                language,
                vad_filter=vad_filter,
                min_silence_duration_ms=min_silence_duration_ms,
                speech_pad_ms=speech_pad_ms
            )
            print(f"    >> Returned {len(raw_segments)} segments", flush=True)
            
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
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    except:
        pass
    
    return results


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

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
    return _transcribe_with_shared_model(
        audio_path,
        model_name,
        compute_type,
        language,
        vad_filter=False  # Don't filter voiceover
    )


def transcribe_voiceover_media(
    media_path: str,
    output_srt_path: str = None,
    model_name: str = "base",
    language: str = None,
    compute_type: str = "auto",
    cache_dir: str = None
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
    
    Returns:
        Path to the generated SRT file
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
        
        # Transcribe the extracted audio
        segments = _transcribe_with_shared_model(
            audio_path,
            model_name,
            compute_type,
            language,
            vad_filter=False  # Don't filter voiceover
        )
        
        # Clean up extracted audio
        try:
            Path(audio_path).unlink()
        except:
            pass
    
    elif media_path.suffix.lower() in audio_extensions:
        # It's already an audio file
        segments = _transcribe_with_shared_model(
            str(media_path),
            model_name,
            compute_type,
            language,
            vad_filter=False
        )
    else:
        raise ValueError(f"Unsupported media format: {media_path.suffix}")
    
    if not segments:
        raise RuntimeError(f"No segments generated from transcription of {media_path}")
    
    # Write SRT file
    _write_srt(segments, str(srt_path))
    
    return str(srt_path)


def _write_srt(segments: List[dict], srt_path: str):
    """Write segments to SRT format."""
    def format_timestamp(seconds: float) -> str:
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, 1):
            start = seg.get('start', 0)
            end = seg.get('end', 0)
            text = seg.get('text', '').strip()
            
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            f.write(f"{text}\n\n")


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