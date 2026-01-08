"""
Unit tests for retry logic.
"""

import pytest
import time
from src.llm_client.retry import with_retry
from src.llm_client.exceptions import LLMTimeoutError, LLMProviderError


class TestWithRetry:
    """Test with_retry function."""

    def test_successful_call_first_try(self):
        """Test function succeeds on first try."""
        call_count = 0

        def func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = with_retry(func, max_retries=3)

        assert result == "success"
        assert call_count == 1

    def test_retry_on_exception(self):
        """Test function retries on exception."""
        call_count = 0

        def func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary error")
            return "success"

        result = with_retry(func, max_retries=3, base_delay=0.01)

        assert result == "success"
        assert call_count == 3

    def test_max_retries_exhausted(self):
        """Test that max retries are exhausted and error is raised."""
        call_count = 0

        def func():
            nonlocal call_count
            call_count += 1
            raise Exception("Persistent error")

        with pytest.raises(LLMProviderError) as exc_info:
            with_retry(func, max_retries=3, base_delay=0.01)

        assert call_count == 3
        assert "failed after 3 attempts" in str(exc_info.value).lower()

    def test_timeout_error_converted(self):
        """Test that timeout errors are converted to LLMTimeoutError."""
        def func():
            raise Exception("Request timeout exceeded")

        with pytest.raises(LLMTimeoutError):
            with_retry(func, max_retries=2, base_delay=0.01)

    def test_exponential_backoff(self):
        """Test exponential backoff delays."""
        call_times = []

        def func():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise Exception("Retry")
            return "success"

        with_retry(func, max_retries=3, base_delay=0.1)

        # Check delays between calls
        if len(call_times) >= 2:
            delay1 = call_times[1] - call_times[0]
            # First delay should be ~0.1s (2^0 * base_delay)
            assert 0.05 < delay1 < 0.2

        if len(call_times) >= 3:
            delay2 = call_times[2] - call_times[1]
            # Second delay should be ~0.2s (2^1 * base_delay)
            assert 0.15 < delay2 < 0.4

    def test_different_max_retries(self):
        """Test different max_retries values."""
        # Test with 1 retry
        call_count = 0

        def func():
            nonlocal call_count
            call_count += 1
            raise Exception("Error")

        with pytest.raises(LLMProviderError):
            with_retry(func, max_retries=1, base_delay=0.01)

        assert call_count == 1

        # Test with 5 retries
        call_count = 0
        with pytest.raises(LLMProviderError):
            with_retry(func, max_retries=5, base_delay=0.01)

        assert call_count == 5

    def test_no_retries_on_success(self):
        """Test that successful calls don't trigger retries."""
        call_count = 0
        start_time = time.time()

        def func():
            nonlocal call_count
            call_count += 1
            return "immediate success"

        result = with_retry(func, max_retries=10, base_delay=1.0)

        elapsed = time.time() - start_time

        assert result == "immediate success"
        assert call_count == 1
        # Should be very fast (no delays)
        assert elapsed < 0.1

    def test_provider_error_message(self):
        """Test that provider error message includes attempt count."""
        def func():
            raise Exception("API key invalid")

        with pytest.raises(LLMProviderError) as exc_info:
            with_retry(func, max_retries=2, base_delay=0.01)

        error_msg = str(exc_info.value)
        assert "2 attempts" in error_msg
        assert "API key invalid" in error_msg

    def test_timeout_in_error_message(self):
        """Test timeout errors are detected correctly."""
        def func():
            raise Exception("Connection timeout after 30 seconds")

        with pytest.raises(LLMTimeoutError):
            with_retry(func, max_retries=1, base_delay=0.01)

        def func2():
            raise Exception("Deadline exceeded")

        with pytest.raises(LLMTimeoutError):
            with_retry(func2, max_retries=1, base_delay=0.01)

    def test_returns_correct_type(self):
        """Test that return types are preserved."""
        # String return
        result = with_retry(lambda: "text", max_retries=1)
        assert isinstance(result, str)

        # Dict return
        result = with_retry(lambda: {"key": "value"}, max_retries=1)
        assert isinstance(result, dict)

        # List return
        result = with_retry(lambda: [1, 2, 3], max_retries=1)
        assert isinstance(result, list)

        # None return
        result = with_retry(lambda: None, max_retries=1)
        assert result is None

    def test_exception_chaining(self):
        """Test that original exception is chained."""
        original_error = ValueError("Original error")

        def func():
            raise original_error

        with pytest.raises(LLMProviderError) as exc_info:
            with_retry(func, max_retries=1, base_delay=0.01)

        # Check exception chain
        assert exc_info.value.__cause__ == original_error
