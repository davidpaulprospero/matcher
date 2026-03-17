"""
Retry logic with exponential backoff for LLM requests.
"""

import time
import logging
from typing import Callable, TypeVar, Any, Optional
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


def _format_correlation(correlation_id: Optional[str]) -> str:
    """Format correlation ID into log prefix."""
    if correlation_id:
        return f"[corr:{correlation_id}]"
    return ""


def with_retry(
    func: Callable[[], T],
    max_retries: int = 3,
    timeout: int = 120,
    base_delay: float = 2.0,
    correlation_id: Optional[str] = None,
    provider: str = "LLM",
    model: str = ""
) -> T:
    """
    Execute a function with exponential backoff retry logic.

    Args:
        func: Function to execute (should take no arguments)
        max_retries: Maximum number of retry attempts
        timeout: Timeout for each attempt (seconds)
        base_delay: Base delay for exponential backoff (seconds)
        correlation_id: Optional correlation ID for request tracing
        provider: Provider name for logging
        model: Model name for logging

    Returns:
        Result of function execution

    Raises:
        LLMTimeoutError: If all retries fail due to timeout
        LLMProviderError: If all retries fail due to provider errors
        Exception: Other exceptions are re-raised
    """
    last_exception = None
    start_time = time.time()
    corr = _format_correlation(correlation_id)
    provider_upper = provider.upper()

    for attempt in range(max_retries):
        attempt_start = time.time()
        try:
            result = func()
            attempt_duration_ms = (time.time() - attempt_start) * 1000
            total_duration_ms = (time.time() - start_time) * 1000

            # Log successful attempt timing at DEBUG level
            logger.debug(
                f"[{provider_upper}] LLM request succeeded{corr}: "
                f"model={model}, attempt={attempt + 1}, duration_ms={attempt_duration_ms:.2f}, "
                f"total_duration_ms={total_duration_ms:.2f}"
            )
            return result

        except Exception as e:
            last_exception = e
            error_str = str(e)

            # Check for permanent errors that shouldn't be retried
            if _is_permanent_error(error_str):
                logger.error(
                    f"[{provider_upper}] LLM request failed (permanent error, not retrying){corr}: "
                    f"model={model}, error={error_str}"
                )
                break  # Exit retry loop immediately

            # Log retry attempt with attempt number and previous failure reason
            if attempt < max_retries - 1:
                wait_time = base_delay ** attempt  # Exponential: 2^0=1s, 2^1=2s, 2^2=4s
                logger.warning(
                    f"[{provider_upper}] LLM request retry{corr}: "
                    f"model={model}, attempt={attempt + 1}/{max_retries}, "
                    f"previous_failure='{error_str[:100]}', retrying in {wait_time:.1f}s..."
                )
                time.sleep(wait_time)
            else:
                logger.error(
                    f"[{provider_upper}] LLM request failed after {max_retries} attempts{corr}: "
                    f"model={model}, last_error='{error_str[:100]}'"
                )

    # All retries exhausted
    error_msg = str(last_exception).lower()
    if "timeout" in error_msg or "deadline" in error_msg:
        raise LLMTimeoutError(f"Request timed out after {max_retries} attempts") from last_exception
    else:
        raise LLMProviderError(f"Request failed after {max_retries} attempts: {str(last_exception)}") from last_exception
