"""
Transcription module with GPU acceleration

Supports two providers:
- faster-whisper (default, 4x faster using CTranslate2)
- openai-whisper (original, fallback)
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import threading

from .config import Config
from .utils import (
    SRTSegment, CacheManager, ProgressBar, 
    write_srt_file, parse_srt_file
)

logger = logging.getLogger(__name__)

# Thread-safe model loading
_model_lock = threading.Lock()
_loaded_model = None
_loaded_model_name = None
_loaded_provider = None


def _detect_compute_type() -> str:
    """Auto-detect best compute type for faster-whisper"""
    try:
        import torch
        if torch.cuda.is_available():
            # Check GPU compute capability
            capability = torch.cuda.get_device_capability()
            if capability[0] >= 7:  # Volta or newer (V100, RTX 20xx+)
                return "float16"
            else:
                return "int8_float16"
        else:
            return "int8"  # CPU fallback
    except:
        return "int8"


def get_faster_whisper_model(model_name: str, compute_type: str = "auto"):
    """Load faster-whisper model"""
    global _loaded_model, _loaded_model_name, _loaded_provider
    
    with _model_lock:
        if (_loaded_model is not None and 
            _loaded_model_name == model_name and 
            _loaded_provider == "faster-whisper"):
            return _loaded_model
        
        from faster_whisper import WhisperModel
        import torch
        
        # Auto-detect compute type
        if compute_type == "auto":
            compute_type = _detect_compute_type()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading faster-whisper '{model_name}' on {device.upper()} (compute: {compute_type})")
        
        _loaded_model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type
        )
        _loaded_model_name = model_name
        _loaded_provider = "faster-whisper"
        
        return _loaded_model


def get_openai_whisper_model(model_name: str, use_gpu: bool = True):
    """Load original OpenAI Whisper model (fallback)"""
    global _loaded_model, _loaded_model_name, _loaded_provider
    
    with _model_lock:
        if (_loaded_model is not None and 
            _loaded_model_name == model_name and 
            _loaded_provider == "openai-whisper"):
            return _loaded_model
        
        import whisper
        import torch
        
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading OpenAI Whisper '{model_name}' on {device.upper()}")
        
        _loaded_model = whisper.load_model(model_name, device=device)
        _loaded_model_name = model_name
        _loaded_provider = "openai-whisper"
        
        return _loaded_model


def get_whisper_model(model_name: str, use_gpu: bool = True):
    """Get or load Whisper model (cached) - legacy compatibility"""
    return get_openai_whisper_model(model_name, use_gpu)


def extract_audio_safe(video_path: str, cache: CacheManager, video_hash: str) -> Optional[str]:
    """
    Extract and re-encode audio from video to handle corrupted audio streams.
    Returns path to the cleaned audio file.
    """
    audio_path = cache.audio_dir / f"{video_hash}.wav"
    
    if audio_path.exists() and audio_path.stat().st_size > 10000:
        return str(audio_path)
    
    logger.debug(f"Extracting audio from {Path(video_path).name}...")
    
    # First, check if video has audio stream
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0",
        video_path
    ]
    
    try:
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        if not probe_result.stdout.strip():
            logger.warning(f"No audio stream in {Path(video_path).name}")
            return None
    except Exception as e:
        logger.debug(f"Audio probe failed: {e}")
    
    # FFmpeg command with GPU-accelerated decoding (CUDA) + error tolerance flags
    cmd = [
        "ffmpeg",
        "-y",
        "-hwaccel", "cuda",  # GPU decoding for NVIDIA
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
        
        if audio_path.exists() and audio_path.stat().st_size > 10000:
            return str(audio_path)
        
        # Fallback to CPU if GPU fails
        cmd_cpu = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-err_detect", "ignore_err",
            "-fflags", "+genpts+igndts",
            str(audio_path)
        ]
        result = subprocess.run(cmd_cpu, capture_output=True, text=True, timeout=300)
        
        if audio_path.exists() and audio_path.stat().st_size > 10000:
            return str(audio_path)
        else:
            logger.warning(f"Audio extraction produced empty/small file for {Path(video_path).name}")
            return None
            
    except subprocess.TimeoutExpired:
        logger.warning(f"Audio extraction timed out for {Path(video_path).name}")
        return None
    except Exception as e:
        logger.warning(f"Audio extraction failed for {Path(video_path).name}: {e}")
        return None


def _transcribe_with_faster_whisper(
    audio_path: str,
    model,
    language: Optional[str] = None
) -> List[dict]:
    """Transcribe using faster-whisper"""
    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=True,  # Voice activity detection for better accuracy
        vad_parameters=dict(
            min_silence_duration_ms=500,
            speech_pad_ms=200
        )
    )
    
    # Convert generator to list of segment dicts
    result_segments = []
    for seg in segments:
        result_segments.append({
            'start': seg.start,
            'end': seg.end,
            'text': seg.text
        })
    
    return result_segments


def _transcribe_with_openai_whisper(
    audio_path: str,
    model,
    language: Optional[str] = None
) -> List[dict]:
    """Transcribe using OpenAI Whisper"""
    result = model.transcribe(
        audio_path,
        language=language,
        verbose=False
    )
    return result.get('segments', [])


def transcribe_single_video(
    video_path: str,
    cache: CacheManager,
    config: Config
) -> Tuple[str, List[SRTSegment]]:
    """
    Transcribe a single video file.
    Returns (video_path, segments)
    """
    # FAST PATH 1: Check if SRT file already exists alongside video
    srt_path = Path(video_path).with_suffix('.srt')
    if srt_path.exists():
        logger.debug(f"Using existing SRT for {Path(video_path).name}")
        try:
            segments = parse_srt_file(str(srt_path))
            # Update source_file to be the video path, not the SRT path
            for seg in segments:
                seg.source_file = video_path
            if segments:
                return video_path, segments
        except Exception as e:
            logger.warning(f"Could not parse existing SRT {srt_path}: {e}")
    
    # FAST PATH 2: Check cache by file hash (fast hash based on path+size+mtime)
    video_hash = cache.get_file_hash(video_path)
    
    # Check cache first
    cached = cache.get_transcription(video_hash)
    if cached:
        logger.debug(f"Using cached transcription for {Path(video_path).name}")
        return video_path, cached
    
    # Extract audio safely
    audio_path = extract_audio_safe(video_path, cache, video_hash)
    if audio_path is None:
        logger.warning(f"Skipping {Path(video_path).name} - could not extract valid audio")
        # Cache empty result to avoid re-trying
        cache.save_transcription(video_hash, [])
        return video_path, []
    
    # Determine provider
    provider = getattr(config.transcription, 'provider', 'faster-whisper')
    compute_type = getattr(config.transcription, 'compute_type', 'auto')
    
    # Try faster-whisper first, fall back to openai-whisper
    result_segments = []
    try:
        if provider == "faster-whisper":
            try:
                model = get_faster_whisper_model(
                    config.transcription.model,
                    compute_type
                )
                result_segments = _transcribe_with_faster_whisper(
                    audio_path,
                    model,
                    config.transcription.language
                )
            except ImportError:
                logger.warning("faster-whisper not installed, falling back to openai-whisper")
                logger.warning("Install with: pip install faster-whisper")
                provider = "openai-whisper"
        
        if provider == "openai-whisper":
            model = get_openai_whisper_model(
                config.transcription.model,
                config.transcription.use_gpu
            )
            result_segments = _transcribe_with_openai_whisper(
                audio_path,
                model,
                config.transcription.language
            )
            
    except RuntimeError as e:
        error_msg = str(e)
        if "reshape" in error_msg or "empty" in error_msg.lower():
            logger.warning(f"Audio data issue for {Path(video_path).name}: corrupt or empty audio")
            try:
                Path(audio_path).unlink()
            except:
                pass
        else:
            logger.warning(f"Transcription failed for {Path(video_path).name}: {e}")
        cache.save_transcription(video_hash, [])
        return video_path, []
    except Exception as e:
        logger.warning(f"Transcription failed for {Path(video_path).name}: {e}")
        cache.save_transcription(video_hash, [])
        return video_path, []
    
    # Convert to segments
    segments = []
    for i, seg in enumerate(result_segments, 1):
        text = seg.get('text', '').strip()
        if text:  # Skip empty segments
            segments.append(SRTSegment(
                index=i,
                start_time=seg['start'],
                end_time=seg['end'],
                text=text,
                source_file=video_path
            ))
    
    # Cache the result
    cache.save_transcription(video_hash, segments)
    
    # Also save as SRT file
    srt_path = Path(video_path).with_suffix('.srt')
    if not srt_path.exists() and segments:
        write_srt_file(segments, str(srt_path))
    
    if not segments:
        logger.warning(f"No speech detected in {Path(video_path).name}")
    
    return video_path, segments


def transcribe_videos_parallel(
    video_paths: List[str],
    cache: CacheManager,
    config: Config,
    progress_callback=None
) -> dict:
    """
    Transcribe multiple videos. Uses GPU if available (sequential but fast).
    Returns dict mapping video_path -> segments
    """
    results = {}
    
    # Pre-check: count how many have existing SRT files or cache
    cached_count = 0
    for vp in video_paths:
        srt_path = Path(vp).with_suffix('.srt')
        if srt_path.exists():
            cached_count += 1
        else:
            video_hash = cache.get_file_hash(vp)
            if cache.get_transcription(video_hash):
                cached_count += 1
    
    need_transcription = len(video_paths) - cached_count
    if cached_count > 0:
        logger.info(f"Found {cached_count}/{len(video_paths)} cached transcriptions")
    
    if need_transcription == 0:
        logger.info("All videos already transcribed - loading from cache...")
    else:
        # Check GPU availability
        import torch
        has_gpu = torch.cuda.is_available()
        use_gpu = config.transcription.use_gpu and has_gpu
        
        provider = getattr(config.transcription, 'provider', 'faster-whisper')
        
        if use_gpu:
            logger.info(f"Transcribing {need_transcription} videos with {provider} (GPU)")
        else:
            if config.transcription.use_gpu and not has_gpu:
                logger.warning("GPU requested but CUDA not available, using CPU")
            logger.info(f"Transcribing {need_transcription} videos with {provider} (CPU)")
    
    progress = ProgressBar(len(video_paths), "Transcribing")
    
    # Always sequential - GPU is fast enough, and parallel CPU has issues
    for video_path in video_paths:
        _, segments = transcribe_single_video(video_path, cache, config)
        results[video_path] = segments
        progress.update(1, Path(video_path).name[:40])
        if progress_callback:
            progress_callback(video_path, segments)
    
    progress.close()
    
    return results


def transcribe_voiceover(
    voiceover_path: str,
    cache: CacheManager,
    config: Config
) -> List[SRTSegment]:
    """
    Transcribe voiceover file (audio or video).
    Returns SRT segments.
    """
    # Check if it's already an SRT
    if voiceover_path.lower().endswith('.srt'):
        return parse_srt_file(voiceover_path)
    
    # Check if SRT already exists
    srt_path = Path(voiceover_path).with_suffix('.srt')
    if srt_path.exists():
        logger.info(f"Found existing voiceover SRT: {srt_path}")
        return parse_srt_file(str(srt_path))
    
    # Transcribe
    logger.info(f"Transcribing voiceover: {Path(voiceover_path).name}")
    _, segments = transcribe_single_video(voiceover_path, cache, config)
    
    # Save SRT
    if segments:
        write_srt_file(segments, str(srt_path))
        logger.info(f"Created voiceover SRT: {srt_path}")
    
    return segments


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe"""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except:
        return 0.0


def release_model():
    """Release loaded model to free memory"""
    global _loaded_model, _loaded_model_name, _loaded_provider
    
    with _model_lock:
        if _loaded_model is not None:
            del _loaded_model
            _loaded_model = None
            _loaded_model_name = None
            _loaded_provider = None
            
            # Clear CUDA cache
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            
            logger.info("Released transcription model from memory")


def transcribe_voiceover_audio(
    audio_path: str,
    output_srt_path: str = None,
    model_name: str = "base",
    language: str = "en",
    compute_type: str = "auto"
) -> str:
    """
    Transcribe a voiceover audio file (MP3, WAV, M4A, etc.) to SRT format.
    Uses faster-whisper with GPU acceleration.
    
    Args:
        audio_path: Path to audio file (MP3, WAV, M4A, FLAC, OGG, etc.)
        output_srt_path: Output SRT path (default: same name with .srt extension)
        model_name: Whisper model name (tiny, base, small, medium, large)
        language: Language code (e.g., 'en', 'ja', 'es')
        compute_type: Compute type (auto, float16, int8_float16, int8)
    
    Returns:
        Path to generated SRT file
    """
    audio_path = Path(audio_path)
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Supported audio formats
    audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac', '.opus'}
    if audio_path.suffix.lower() not in audio_extensions:
        raise ValueError(f"Unsupported audio format: {audio_path.suffix}")
    
    # Determine output path
    if output_srt_path is None:
        output_srt_path = audio_path.with_suffix('.srt')
    else:
        output_srt_path = Path(output_srt_path)
    
    logger.info(f"Transcribing voiceover: {audio_path.name}")
    logger.info(f"Output SRT: {output_srt_path}")
    
    # Convert to WAV if needed (faster-whisper works best with WAV)
    temp_wav = None
    transcribe_path = str(audio_path)
    
    if audio_path.suffix.lower() != '.wav':
        temp_wav = audio_path.with_suffix('.temp.wav')
        logger.info(f"Converting to WAV for transcription...")
        
        # Use FFmpeg with CUDA decoding if available
        cmd = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(temp_wav)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if temp_wav.exists():
                transcribe_path = str(temp_wav)
            else:
                raise RuntimeError("WAV conversion failed")
        except Exception as e:
            logger.error(f"Failed to convert audio: {e}")
            raise
    
    # Load model with GPU
    try:
        model = get_faster_whisper_model(model_name, compute_type)
    except ImportError:
        logger.error("faster-whisper not installed. Install with: pip install faster-whisper")
        raise
    
    # Transcribe
    logger.info(f"Running transcription with faster-whisper ({model_name})...")
    
    try:
        segments_raw, info = model.transcribe(
            transcribe_path,
            language=language,
            beam_size=5,
            word_timestamps=False,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200
            )
        )
        
        # Convert to SRT format
        segments = list(segments_raw)
        logger.info(f"Detected language: {info.language} (prob: {info.language_probability:.2f})")
        logger.info(f"Transcribed {len(segments)} segments")
        
        # Write SRT file
        import srt
        from datetime import timedelta
        
        srt_segments = []
        for i, seg in enumerate(segments, 1):
            srt_segments.append(srt.Subtitle(
                index=i,
                start=timedelta(seconds=seg.start),
                end=timedelta(seconds=seg.end),
                content=seg.text.strip()
            ))
        
        with open(output_srt_path, 'w', encoding='utf-8') as f:
            f.write(srt.compose(srt_segments))
        
        logger.info(f"✓ Saved SRT: {output_srt_path}")
        
    finally:
        # Cleanup temp WAV
        if temp_wav and temp_wav.exists():
            try:
                temp_wav.unlink()
            except:
                pass
    
    return str(output_srt_path)
