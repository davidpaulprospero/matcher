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
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
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


def get_available_gpu_memory() -> float:
    """
    Get available GPU memory in MB (US-60-007).

    Queries CUDA device for total memory and subtracts currently allocated memory
    to determine how much is available for new allocations.

    Returns:
        Available GPU memory in MB, or 0.0 if CUDA unavailable
    """
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            # Account for some reserved memory overhead
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            # Available = total - max(allocated, reserved) to be conservative
            used = max(allocated, reserved)
            available = total - used
            return available
    except ImportError:
        pass
    return 0.0


# Model size estimates in MB (approximate VRAM at float16)
# Used for automatic model downgrade when GPU memory is insufficient
MODEL_MEMORY_REQUIREMENTS = {
    "large-v3": 3000,
    "large-v2": 3000,
    "large": 3000,
    "medium": 2000,
    "small": 1000,
    "base": 500,
    "tiny": 400,
}

# Model downgrade chain (larger -> smaller)
MODEL_DOWNGRADE_ORDER = ["large-v3", "large-v2", "large", "medium", "small", "base", "tiny"]


def _select_model_for_memory(requested_model: str, available_memory_mb: float, min_memory_mb: float) -> str:
    """
    Select appropriate model based on available GPU memory (US-60-007).

    If requested model requires more memory than available, automatically
    downgrades to a smaller model that fits within memory constraints.

    Args:
        requested_model: Model name requested by user
        available_memory_mb: Available GPU memory in MB
        min_memory_mb: Minimum memory threshold from config

    Returns:
        Model name to use (may be downgraded from requested)
    """
    # If we have plenty of memory, use requested model
    if available_memory_mb >= min_memory_mb:
        model_req = MODEL_MEMORY_REQUIREMENTS.get(requested_model, 500)
        if available_memory_mb >= model_req:
            return requested_model

    # Find position of requested model in downgrade chain
    try:
        start_idx = MODEL_DOWNGRADE_ORDER.index(requested_model)
    except ValueError:
        # Unknown model, assume it's small enough
        start_idx = len(MODEL_DOWNGRADE_ORDER) - 1

    # Try each model from requested downward
    for model in MODEL_DOWNGRADE_ORDER[start_idx:]:
        model_req = MODEL_MEMORY_REQUIREMENTS.get(model, 500)
        if available_memory_mb >= model_req:
            if model != requested_model:
                logger.warning(
                    f"Insufficient GPU memory ({available_memory_mb:.0f}MB available). "
                    f"Downgrading model from '{requested_model}' to '{model}'"
                )
            return model

    # If nothing fits, return tiny as last resort
    logger.warning(
        f"Very low GPU memory ({available_memory_mb:.0f}MB). Using 'tiny' model."
    )
    return "tiny"


class WhisperClient:
    """
    Thread-safe Whisper model client with GPU locking.

    Manages a shared WhisperModel instance for efficient GPU usage
    in parallel transcription scenarios.
    """

    def __init__(
        self,
        model_name: str = "base",
        compute_type: str = "auto",
        minimum_gpu_memory_mb: int = 2000,
        auto_downgrade_model: bool = True,
        gpu_transcription_timeout: int = 300,
        num_workers: int = 1,
        cpu_threads: int = 4
    ):
        """
        Initialize Whisper client with model configuration.

        Args:
            model_name: Whisper model name (base, small, medium, large, etc.)
            compute_type: Compute type (auto, float16, int8)
            minimum_gpu_memory_mb: Minimum GPU memory required (US-60-007)
            auto_downgrade_model: Automatically downgrade model if insufficient memory
            gpu_transcription_timeout: Max seconds for a single transcribe() call (US-79-002)
            num_workers: Workers for WhisperModel batched decoding (US-79-007)
            cpu_threads: CPU threads for ctranslate2 operations (US-79-007)
        """
        self.model_name = model_name
        self.compute_type = compute_type
        self.minimum_gpu_memory_mb = minimum_gpu_memory_mb
        self.auto_downgrade_model = auto_downgrade_model
        self.gpu_transcription_timeout = gpu_transcription_timeout
        self.num_workers = num_workers
        self.cpu_threads = cpu_threads

    def get_model(self):
        """
        Get or create the shared WhisperModel instance.

        Thread-safe initialization with double-checked locking.
        Performs GPU memory pre-check and automatic model downgrade if needed (US-60-007).

        Returns:
            WhisperModel instance
        """
        global _shared_model, _model_config

        # Determine actual model to use (may be downgraded based on GPU memory)
        actual_model = self.model_name
        if self.auto_downgrade_model:
            available_memory = get_available_gpu_memory()
            if available_memory > 0:  # Only check if CUDA available
                actual_model = _select_model_for_memory(
                    self.model_name,
                    available_memory,
                    self.minimum_gpu_memory_mb
                )
            elif available_memory == 0:
                # CUDA not available - warn if below threshold
                logger.info("CUDA not available - GPU memory check skipped")

        # Check if we need to (re)initialize
        current_config = {"model": actual_model, "compute_type": self.compute_type}

        if _shared_model is not None and _model_config == current_config:
            return _shared_model

        with _gpu_lock:
            # Double-check after acquiring lock
            if _shared_model is not None and _model_config == current_config:
                return _shared_model

            # Log GPU memory before initialization
            mem_before_alloc, mem_before_reserved = _get_gpu_memory_mb()
            available_mem = get_available_gpu_memory()
            logger.info(f"GPU memory before model init: allocated={mem_before_alloc:.1f}MB, reserved={mem_before_reserved:.1f}MB, available={available_mem:.1f}MB")

            # Warn if below threshold (US-60-007)
            if available_mem > 0 and available_mem < self.minimum_gpu_memory_mb:
                logger.warning(
                    f"GPU memory ({available_mem:.0f}MB) below threshold ({self.minimum_gpu_memory_mb}MB). "
                    f"Performance may be degraded."
                )

            logger.info(f"Initializing WhisperModel...")
            logger.info(f"  Model: {actual_model}" + (f" (downgraded from {self.model_name})" if actual_model != self.model_name else ""))
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
                    actual_model,
                    device=device,
                    compute_type=actual_compute,
                    num_workers=self.num_workers,
                    cpu_threads=self.cpu_threads
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
                # Run model.transcribe() with timeout guard (US-79-002)
                # GPU calls can hang indefinitely; this prevents blocking the pipeline
                start_time = time.monotonic()

                def _do_transcribe():
                    return model.transcribe(
                        audio_path,
                        language=language,
                        vad_filter=vad_filter,
                        vad_parameters=dict(
                            min_silence_duration_ms=min_silence_duration_ms,
                            speech_pad_ms=speech_pad_ms
                        ),
                        word_timestamps=word_timestamps
                    )

                timeout = self.gpu_transcription_timeout
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_do_transcribe)
                    try:
                        segments, info = future.result(timeout=timeout)
                    except FuturesTimeoutError:
                        elapsed = time.monotonic() - start_time
                        logger.warning(
                            f"GPU transcription timeout for {audio_name}: "
                            f"elapsed={elapsed:.1f}s, timeout={timeout}s"
                        )
                        from src.transcription.exceptions import TransientTranscriptionError
                        raise TransientTranscriptionError(
                            f"GPU transcription timed out after {elapsed:.1f}s "
                            f"(limit: {timeout}s) for {audio_name}"
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
                # Re-raise TransientTranscriptionError without catching it
                from src.transcription.exceptions import TransientTranscriptionError
                if isinstance(e, TransientTranscriptionError):
                    raise
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
