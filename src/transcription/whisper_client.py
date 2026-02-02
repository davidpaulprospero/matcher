"""
Whisper model client with GPU lock management.

Provides thread-safe access to shared WhisperModel for parallel transcription.

CRITICAL FIX:
The _lock error was caused by multiple ThreadPoolExecutor workers trying to
simultaneously access the GPU through separate WhisperModel instances.
ctranslate2 (used by faster-whisper) cannot handle concurrent GPU access.

SOLUTION:
- Use a SINGLE shared WhisperModel instance
- Protect GPU transcription with a threading.Lock() mutex
- Parallelize only the CPU-bound work (audio extraction, I/O)
- Serialize GPU transcription through the shared model
"""

import logging
import sys
import threading
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Global shared model and GPU mutex
_gpu_lock = threading.RLock()  # RLock is REENTRANT - allows same thread to acquire multiple times
_shared_model = None  # Shared WhisperModel instance
_model_config = {}  # Model configuration cache


def _get_gpu_memory_mb() -> tuple[float, float]:
    """
    Get current GPU memory usage.

    Returns:
        Tuple of (allocated_mb, reserved_mb), or (0.0, 0.0) if CUDA unavailable
    """
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            return allocated, reserved
    except ImportError:
        pass
    return 0.0, 0.0


class WhisperClient:
    """
    Thread-safe Whisper model client with GPU locking.

    Manages a shared WhisperModel instance for efficient GPU usage
    in parallel transcription scenarios.
    """

    def __init__(self, model_name: str = "base", compute_type: str = "auto"):
        """
        Initialize Whisper client with model configuration.

        Args:
            model_name: Whisper model name (base, small, medium, large, etc.)
            compute_type: Compute type (auto, float16, int8)
        """
        self.model_name = model_name
        self.compute_type = compute_type

    def get_model(self):
        """
        Get or create the shared WhisperModel instance.

        Thread-safe initialization with double-checked locking.

        Returns:
            WhisperModel instance
        """
        global _shared_model, _model_config

        # Check if we need to (re)initialize
        current_config = {"model": self.model_name, "compute_type": self.compute_type}

        if _shared_model is not None and _model_config == current_config:
            return _shared_model

        with _gpu_lock:
            # Double-check after acquiring lock
            if _shared_model is not None and _model_config == current_config:
                return _shared_model

            # Log GPU memory before initialization
            mem_before_alloc, mem_before_reserved = _get_gpu_memory_mb()
            logger.info(f"GPU memory before model init: allocated={mem_before_alloc:.1f}MB, reserved={mem_before_reserved:.1f}MB")

            logger.info(f"Initializing WhisperModel...")
            logger.info(f"  Model: {self.model_name}")
            logger.info(f"  Compute type: {self.compute_type}")

            try:
                logger.debug("Step 1: Importing faster_whisper...")
                from faster_whisper import WhisperModel
                logger.debug("Step 1: Done")

                # Determine device
                device = "cuda"
                actual_compute = self.compute_type

                if self.compute_type == "auto" or self.compute_type == "int8":
                    logger.debug("Step 2: Checking CUDA availability...")
                    try:
                        import torch
                        cuda_available = torch.cuda.is_available()
                        logger.debug(f"Step 2: CUDA available = {cuda_available}")
                        if cuda_available:
                            device = "cuda"
                            actual_compute = "float16"
                            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                            logger.debug(f"Step 2: GPU memory = {gpu_mem:.1f} GB")
                        else:
                            device = "cpu"
                            actual_compute = "int8"
                    except ImportError:
                        logger.debug("Step 2: torch not available, using CPU")
                        device = "cpu"
                        actual_compute = "int8"

                logger.info(f"Step 3: Creating WhisperModel on {device} ({actual_compute})...")
                logger.info("(This may take 30-60 seconds on first run to download model)")

                sys.stdout.flush()
                sys.stderr.flush()

                _shared_model = WhisperModel(
                    self.model_name,
                    device=device,
                    compute_type=actual_compute,
                    num_workers=1,
                    cpu_threads=4
                )
                _model_config = current_config

                logger.info(f"Step 3: Done!")
                logger.info(f"✓ Model ready on {device} ({actual_compute})")

                # Log GPU memory after initialization
                mem_after_alloc, mem_after_reserved = _get_gpu_memory_mb()
                logger.info(f"GPU memory after model init: allocated={mem_after_alloc:.1f}MB, reserved={mem_after_reserved:.1f}MB")
                mem_delta = mem_after_alloc - mem_before_alloc
                logger.info(f"GPU memory delta from model init: {mem_delta:.1f}MB")

                return _shared_model

            except Exception as e:
                logger.error(f"✗ Failed to load model: {e}")
                import traceback
                traceback.print_exc()
                raise

    def transcribe(
        self,
        audio_path: str,
        language: str = None,
        vad_filter: bool = False,  # Default False - VAD too aggressive for YouTube
        min_silence_duration_ms: int = 200,
        speech_pad_ms: int = 10,
        word_timestamps: bool = False
    ) -> List[dict]:
        """
        Transcribe audio file with GPU lock.

        Acquires GPU lock before transcription, releases after.

        Args:
            audio_path: Path to audio file
            language: Language code or None for auto-detect
            vad_filter: Whether to apply Voice Activity Detection
            min_silence_duration_ms: Minimum silence duration to split segments
            speech_pad_ms: Padding around detected speech
            word_timestamps: Whether to include word-level timestamps

        Returns:
            List of segment dicts with 'start', 'end', 'text', and optionally 'words'
        """
        logger.debug(f"Acquiring GPU lock for {Path(audio_path).name[:40]}...")

        with _gpu_lock:
            logger.debug("Lock acquired, getting model...")
            model = self.get_model()

            audio_name = Path(audio_path).name[:40]
            logger.debug(f"Starting transcription of {audio_name}...")

            try:
                segments, info = model.transcribe(
                    audio_path,
                    language=language,
                    vad_filter=vad_filter,
                    vad_parameters=dict(
                        min_silence_duration_ms=min_silence_duration_ms,
                        speech_pad_ms=speech_pad_ms
                    ),
                    word_timestamps=word_timestamps
                )

                logger.debug("Transcription done, processing segments...")
                result = []
                for seg in segments:
                    seg_data = {
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text.strip()
                    }
                    # Include word-level timestamps if available
                    if word_timestamps and hasattr(seg, 'words') and seg.words:
                        seg_data["words"] = [
                            {"word": w.word, "start": w.start, "end": w.end}
                            for w in seg.words
                        ]
                    result.append(seg_data)

                # Verbose logging for transcription result
                total_duration = sum(s.get('end', 0) - s.get('start', 0) for s in result)
                logger.debug(f"Transcription complete: {audio_name}")
                logger.debug(f"  Segments: {len(result)}, Total duration: {total_duration:.1f}s")
                if result:
                    logger.debug(f"  First segment: '{result[0].get('text', '')[:50]}...'")
                    logger.debug(f"  Word timestamps: {'yes' if result[0].get('words') else 'no'}")

                logger.debug(f"Done: {len(result)} segments")
                return result

            except Exception as e:
                logger.error(f"Transcription error: {e}")
                import traceback
                traceback.print_exc()
                return []

    def cleanup(self):
        """
        Unload the shared whisper model and free GPU memory.

        Call this after batch transcription is complete to reclaim memory
        for subsequent pipeline stages (embedding, matching).
        """
        global _shared_model, _model_config
        import gc

        with _gpu_lock:
            if _shared_model is not None:
                # Log GPU memory before cleanup
                mem_before_alloc, mem_before_reserved = _get_gpu_memory_mb()
                logger.info(f"GPU memory before cleanup: allocated={mem_before_alloc:.1f}MB, reserved={mem_before_reserved:.1f}MB")

                logger.info("Unloading transcription model to free memory...")
                del _shared_model
                _shared_model = None
                _model_config = {}

                # Force garbage collection
                gc.collect()

                # Clear CUDA cache if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        logger.debug("Cleared CUDA cache")
                except ImportError:
                    pass

                # Log GPU memory after cleanup with delta
                mem_after_alloc, mem_after_reserved = _get_gpu_memory_mb()
                mem_delta = mem_before_alloc - mem_after_alloc
                logger.info(f"GPU memory after cleanup: allocated={mem_after_alloc:.1f}MB, reserved={mem_after_reserved:.1f}MB")
                logger.info(f"GPU memory freed by cleanup: {mem_delta:.1f}MB")

                logger.info("Transcription model unloaded")


# Module-level cleanup function for backward compatibility
def cleanup_model():
    """
    Unload the shared whisper model and free GPU memory.

    Backward-compatible function that wraps WhisperClient.cleanup().
    """
    client = WhisperClient()
    client.cleanup()
