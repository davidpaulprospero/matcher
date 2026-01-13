"""
Retry logic with exponential backoff for LLM requests.
"""

import time
import logging
from typing import Callable, TypeVar, Any
from .exceptions import LLMTimeoutError, LLMProviderError

logger = logging.getLogger(__name__)

T = TypeVar('T')

# Errors that are permanent and should not be retried
PERMANENT_ERROR_PATTERNS = [
    "model",  # Model not found, invalid model
    "not found",
    "api key",
    "api_key",
    "authentication",
    "unauthorized",
    "invalid key",
    "permission denied",
]


def _is_permanent_error(error_msg: str) -> bool:
    """Check if an error is permanent and should not be retried."""
    error_lower = error_msg.lower()
    return any(pattern in error_lower for pattern in PERMANENT_ERROR_PATTERNS)


def with_retry(
    func: Callable[[], T],
    max_retries: int = 3,
    timeout: int = 120,
    base_delay: float = 2.0
) -> T:
    """
    Execute a function with exponential backoff retry logic.

    Args:
        func: Function to execute (should take no arguments)
        max_retries: Maximum number of retry attempts
        timeout: Timeout for each attempt (seconds)
        base_delay: Base delay for exponential backoff (seconds)

    Returns:
        Result of function execution

    Raises:
        LLMTimeoutError: If all retries fail due to timeout
        LLMProviderError: If all retries fail due to provider errors
        Exception: Other exceptions are re-raised
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            return func()

        except Exception as e:
            last_exception = e
            error_str = str(e)

            # Check for permanent errors that shouldn't be retried
            if _is_permanent_error(error_str):
                logger.error(f"LLM request failed (permanent error, not retrying): {error_str}")
                break  # Exit retry loop immediately

            # Log the error
            if attempt < max_retries - 1:
                wait_time = base_delay ** attempt  # Exponential: 2^0=1s, 2^1=2s, 2^2=4s
                logger.warning(
                    f"LLM request failed (attempt {attempt + 1}/{max_retries}): {error_str}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                time.sleep(wait_time)
            else:
                logger.error(
                    f"LLM request failed after {max_retries} attempts: {error_str}"
                )

    # All retries exhausted
    error_msg = str(last_exception).lower()
    if "timeout" in error_msg or "deadline" in error_msg:
        raise LLMTimeoutError(f"Request timed out after {max_retries} attempts") from last_exception
    else:
        raise LLMProviderError(f"Request failed after {max_retries} attempts: {str(last_exception)}") from last_exception
