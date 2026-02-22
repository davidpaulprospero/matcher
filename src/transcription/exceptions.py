"""
Transcription exception classes.

Provides exception hierarchy for transcription-related errors, enabling
retry logic to distinguish between transient and permanent errors.
"""

import logging

logger = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """Base class for transcription errors."""
    pass


class TransientTranscriptionError(TranscriptionError):
    """
    Transient transcription error that may succeed on retry.

    Examples:
        - GPU memory pressure (CUDA out of memory)
        - Temporary device unavailability
        - Network interruption during model download
    """
    pass


class PermanentTranscriptionError(TranscriptionError):
    """
    Permanent transcription error that will not succeed on retry.

    Examples:
        - File not found
        - Permission denied
        - Invalid audio format
        - Corrupted file
    """
    pass


class GPUError(TranscriptionError):
    """
    GPU-specific error during transcription (US-110-004).

    Examples:
        - CUDA out of memory
        - GPU not available
        - cuDNN initialization failure
        - CUDA driver error

    This exception triggers automatic fallback to CPU compute type.
    """
    pass


class QualityGateError(TranscriptionError):
    """
    Transcription quality gate failure (US-137-005).

    Raised when transcription fails to meet the minimum quality threshold
    even after retries. The transcription quality (avg_word_confidence)
    is below the configured min_quality_threshold.

    Attributes:
        avg_confidence: The average word confidence of the transcription
        min_threshold: The minimum required quality threshold
        segment_details: List of segment-level quality details
    """
    def __init__(
        self,
        message: str,
        avg_confidence: float = 0.0,
        min_threshold: float = 0.5,
        segment_details: list = None
    ):
        super().__init__(message)
        self.avg_confidence = avg_confidence
        self.min_threshold = min_threshold
        self.segment_details = segment_details or []


# Error patterns that indicate transient errors (worth retrying)
# US-124-007: Enhanced with more patterns from common/error_patterns.py
TRANSIENT_ERROR_PATTERNS = [
    # GPU/CUDA errors (US-124-007: explicit CUDA OOM classification)
    "cuda",
    "out of memory",
    "outofmemoryerror",
    "cudnn",
    "gpu",
    "torch._C._cuda",
    "CUDA error",
    "cudaerror",
    "memory",            # General memory issues
    "device",            # Device unavailability

    # ctranslate2 runtime errors
    "ctranslate2",

    # Timeout errors (US-124-007: network timeouts)
    "timeout",
    "timed out",
    "deadline exceeded",
    "connection timed out",

    # Network-related transient errors (US-124-007: added from common/error_patterns.py)
    "network",
    "connection",
    "dns",
    "resolve",
    "unreachable",
    "refused",
    "reset",
    "broken pipe",
    "http error",
    "ssl",
    "certificate",
    "socket",
    "eof",
    "getaddrinfo",

    # Resource/availability
    "resource",          # Resource unavailable
    "temporary",         # Temporary failures
    "unavailable",       # Service temporarily unavailable
    "service unavailable",
    "retry",
    "backoff",
]

# Error patterns that indicate retry with smaller batch (CUDA OOM specific)
# US-124-007: Classify CUDA out-of-memory as transient for smaller batch retry
CUDA_OOM_PATTERNS = [
    "cuda",
    "out of memory",
    "outofmemoryerror",
    "cudnn",
    "torch._C._cuda",
    "CUDA error",
    "cudaerror",
]

# Error patterns that indicate GPU-specific errors (trigger CPU fallback)
GPU_ERROR_PATTERNS = [
    "cuda",
    "out of memory",
    "cudnn",
    "gpu",
    "torch._C._cuda",
    "CUDA error",
    "cudaerror",
]

# Error patterns that indicate permanent errors (do NOT retry)
PERMANENT_ERROR_PATTERNS = [
    "file not found",
    "no such file",
    "filenotfounderror",
    "permission denied",
    "permissionerror",
    "access denied",
    "invalid audio",
    "unsupported format",
    "corrupt",
    "cannot open",
    "does not exist",
]


def is_transient_error(error: Exception) -> bool:
    """
    Check if an error is transient and worth retrying.

    US-124-007: Added logging for error classification decisions.

    Args:
        error: The exception to check

    Returns:
        True if the error is likely transient, False otherwise
    """
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()

    # Check for permanent errors first (take precedence)
    if isinstance(error, (FileNotFoundError, PermissionError)):
        logger.debug(
            "Error classification: %s (%s) - PERMANENT (builtin type)",
            error_type, error_str[:100]
        )
        return False

    for pattern in PERMANENT_ERROR_PATTERNS:
        if pattern in error_str or pattern in error_type:
            logger.debug(
                "Error classification: %s (%s) - PERMANENT (matched pattern: '%s')",
                error_type, error_str[:100], pattern
            )
            return False

    # Check for transient error patterns
    for pattern in TRANSIENT_ERROR_PATTERNS:
        if pattern in error_str or pattern in error_type:
            # US-124-007: Log CUDA OOM specifically for smaller batch retry
            if pattern in CUDA_OOM_PATTERNS or "out of memory" in pattern:
                logger.info(
                    "Error classification: %s (%s) - TRANSIENT (CUDA OOM, retry with smaller batch) "
                    "matched pattern: '%s'",
                    error_type, error_str[:100], pattern
                )
            else:
                logger.debug(
                    "Error classification: %s (%s) - TRANSIENT (matched pattern: '%s')",
                    error_type, error_str[:100], pattern
                )
            return True

    # Default: treat unknown errors as non-transient (don't retry)
    # US-124-007: Log unknown errors for debugging
    logger.debug(
        "Error classification: %s (%s) - UNKNOWN (defaulting to non-transient)",
        error_type, error_str[:100]
    )
    return False


def is_cuda_oom_error(error: Exception) -> bool:
    """
    Check if an error is a CUDA out-of-memory error (US-124-007).

    Used to determine if transcription should be retried with a smaller batch size.

    Args:
        error: The exception to check

    Returns:
        True if the error is CUDA OOM, False otherwise
    """
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()

    for pattern in CUDA_OOM_PATTERNS:
        if pattern in error_str or pattern in error_type:
            logger.debug(
                "CUDA OOM detection: %s matched pattern '%s'",
                error_str[:100], pattern
            )
            return True

    return False


def is_gpu_error(error: Exception) -> bool:
    """
    Check if an error is GPU-specific and should trigger CPU fallback (US-110-004).

    Args:
        error: The exception to check

    Returns:
        True if the error is GPU-related, False otherwise
    """
    # Check for GPUError exception type first
    if isinstance(error, GPUError):
        return True

    error_str = str(error).lower()
    error_type = type(error).__name__.lower()

    # Check for GPU-specific error patterns
    for pattern in GPU_ERROR_PATTERNS:
        if pattern in error_str or pattern in error_type:
            return True

    return False
