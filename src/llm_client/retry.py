"""
Retry logic with exponential backoff for LLM requests.
"""

import time
import logging
from typing import Callable, TypeVar, Any
from .exceptions import LLMTimeoutError, LLMProviderError

logger = logging.getLogger(__name__)

T = TypeVar('T')


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

            # Log the error
            if attempt < max_retries - 1:
                wait_time = base_delay ** attempt  # Exponential: 2^0=1s, 2^1=2s, 2^2=4s
                logger.warning(
                    f"LLM request failed (attempt {attempt + 1}/{max_retries}): {str(e)}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                time.sleep(wait_time)
            else:
                logger.error(
                    f"LLM request failed after {max_retries} attempts: {str(e)}"
                )

    # All retries exhausted
    error_msg = str(last_exception).lower()
    if "timeout" in error_msg or "deadline" in error_msg:
        raise LLMTimeoutError(f"Request timed out after {max_retries} attempts") from last_exception
    else:
        raise LLMProviderError(f"Request failed after {max_retries} attempts: {str(last_exception)}") from last_exception
