"""
Transcription exception classes.

Provides exception hierarchy for transcription-related errors, enabling
retry logic to distinguish between transient and permanent errors.
"""


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


# Error patterns that indicate transient errors (worth retrying)
TRANSIENT_ERROR_PATTERNS = [
    "cuda",              # CUDA memory errors
    "out of memory",     # GPU/system memory
    "cudnn",             # cuDNN library errors
    "memory",            # General memory issues
    "device",            # Device unavailability
    "ctranslate2",       # ctranslate2 runtime errors
    "timeout",           # Timeout errors
    "resource",          # Resource unavailable
    "temporary",         # Temporary failures
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

    Args:
        error: The exception to check

    Returns:
        True if the error is likely transient, False otherwise
    """
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()

    # Check for permanent errors first (take precedence)
    if isinstance(error, (FileNotFoundError, PermissionError)):
        return False

    for pattern in PERMANENT_ERROR_PATTERNS:
        if pattern in error_str or pattern in error_type:
            return False

    # Check for transient error patterns
    for pattern in TRANSIENT_ERROR_PATTERNS:
        if pattern in error_str or pattern in error_type:
            return True

    # Default: treat unknown errors as non-transient (don't retry)
    return False
