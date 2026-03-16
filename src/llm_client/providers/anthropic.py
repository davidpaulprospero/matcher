"""
Anthropic Claude LLM client implementation.
"""

import logging
import time
from typing import Optional
from ..base import LLMClient, LLMRequest
from ..exceptions import LLMProviderError, LLMTimeoutError
from ..cost import calculate_llm_cost
from src.logging_templates import log_error_with_context

logger = logging.getLogger(__name__)


def _format_correlation(correlation_id: Optional[str]) -> str:
    """Format correlation ID into log prefix."""
    if correlation_id:
        return f"[corr:{correlation_id}]"
    return ""


class AnthropicClient(LLMClient):
    """
    Client for Anthropic Claude models.

    Supports:
    - Text generation
    - System prompts
    - Configurable models
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-haiku-20240307",
        cache_dir: str = ".cache/llm_responses",
        cache_ttl_hours: int = 24,
        cache_skip_low_quality: bool = False
    ):
        """
        Initialize Anthropic client.

        Args:
            api_key: Anthropic API key
            model: Model name (e.g., "claude-3-haiku-20240307", "claude-3-5-sonnet-20241022")
            cache_dir: Cache directory
            cache_ttl_hours: TTL for cached LLM responses in hours (0 = never expire)
            cache_skip_low_quality: If True, skip cache entries with quality_tier='low'
        """
        super().__init__(api_key, model, cache_dir, cache_ttl_hours, cache_skip_low_quality)

        try:
            import anthropic
            import httpx
            self.client = anthropic.Anthropic(
                api_key=api_key,
                timeout=httpx.Timeout(120, connect=30.0)
            )
        except ImportError:
            raise LLMProviderError(
                "anthropic package not installed. "
                "Install with: pip install anthropic"
            )
        except Exception as e:
            raise LLMProviderError(f"Failed to initialize Anthropic client: {e}")

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def _call_api(self, request: LLMRequest) -> str:
        """
        Call Anthropic API.

        Args:
            request: LLM request

        Returns:
            Raw text response

        Raises:
            LLMProviderError: On API errors
            LLMTimeoutError: On timeout
        """
        start_time = time.time()
        prompt_length = len(request.prompt)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Get correlation ID
        correlation_id = getattr(request, 'correlation_id', None)
        corr = _format_correlation(correlation_id)

        # Log request start at INFO level with provider, model, prompt length, timestamp
        logger.info(
            f"[ANTHROPIC] LLM API request{corr}: "
            f"model={self.model}, prompt_length={prompt_length}, "
            f"max_tokens={request.max_tokens}, timestamp={timestamp}"
        )

        try:
            # Build messages
            messages = [{"role": "user", "content": request.prompt}]

            # Build request parameters
            params = {
                "model": self.model,
                "max_tokens": request.max_tokens,
                "messages": messages
            }

            # Add system prompt if provided
            if request.system_prompt:
                params["system"] = request.system_prompt

            # Add temperature if not default
            if request.temperature != 1.0:
                params["temperature"] = request.temperature

            # Log request details at DEBUG level (truncated)
            system_prompt_preview = request.system_prompt[:100] + "..." if request.system_prompt and len(request.system_prompt) > 100 else request.system_prompt
            logger.debug(
                f"[ANTHROPIC] API request{corr}: model={self.model}, "
                f"max_tokens={request.max_tokens}, has_system_prompt={bool(request.system_prompt)}, "
                f"system_prompt={system_prompt_preview}"
            )

            # Make API call
            response = self.client.messages.create(**params)

            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            logger.info(
                f"[ANTHROPIC] LLM API response{corr}: "
                f"model={self.model}, status=success, latency_ms={response_time_ms:.1f}"
            )

            # Extract token usage for cost tracking (US-162-010)
            # Store in response metadata for the generate() method to include
            self._last_tokens_used = None
            self._last_prompt_tokens = None
            self._last_completion_tokens = None

            if hasattr(response, 'usage') and response.usage:
                self._last_prompt_tokens = getattr(response.usage, 'input_tokens', None)
                self._last_completion_tokens = getattr(response.usage, 'output_tokens', None)
                if self._last_prompt_tokens and self._last_completion_tokens:
                    self._last_tokens_used = self._last_prompt_tokens + self._last_completion_tokens
                elif self._last_prompt_tokens:
                    self._last_tokens_used = self._last_prompt_tokens
                elif self._last_completion_tokens:
                    self._last_tokens_used = self._last_completion_tokens

            # Log token usage and cost estimate at INFO level
            if self._last_tokens_used:
                cost = calculate_llm_cost(
                    provider="anthropic",
                    model=self.model,
                    input_tokens=self._last_prompt_tokens,
                    output_tokens=self._last_completion_tokens
                )
                logger.info(
                    f"[ANTHROPIC] Token usage{corr}: tokens_used={self._last_tokens_used}, "
                    f"input={self._last_prompt_tokens}, output={self._last_completion_tokens}, "
                    f"estimated_cost=${cost:.6f}, latency_ms={response_time_ms:.1f}"
                )

            # Extract text from response
            if response.content and len(response.content) > 0:
                result = response.content[0].text
                logger.debug(f"[ANTHROPIC] Response extracted{corr}: {len(result)} chars")
                return result
            else:
                logger.warning(f"[ANTHROPIC] Response has no content{corr}: {response}")
                return ""

        except Exception as e:
            error_msg = str(e).lower()

            # Check for timeout
            if "timeout" in error_msg or "timed out" in error_msg:
                log_error_with_context(logger, "MATCH-005", f"Anthropic request timed out: {e}",
                                       correlation_id=correlation_id,
                                       model=self.model, prompt_length=prompt_length)
                raise LLMTimeoutError(f"Anthropic request timed out: {e}")

            # Check for common API errors
            if "api_key" in error_msg or "authentication" in error_msg or "unauthorized" in error_msg:
                log_error_with_context(logger, "MATCH-004", f"Anthropic authentication error: {e}",
                                       correlation_id=correlation_id,
                                       model=self.model)
                raise LLMProviderError(f"Anthropic authentication error: {e}")
            elif "quota" in error_msg or "rate" in error_msg or "overloaded" in error_msg:
                # Log rate limit at WARNING level for detection/tracking
                logger.warning(f"[MATCH-006] Anthropic rate limit detected{corr}: {e}")
                log_error_with_context(logger, "MATCH-006", f"Anthropic quota/rate limit error: {e}",
                                       correlation_id=correlation_id,
                                       model=self.model, prompt_length=prompt_length)
                raise LLMProviderError(f"Anthropic quota/rate limit error: {e}")
            else:
                log_error_with_context(logger, "MATCH-001", f"Anthropic API error: {e}",
                                       correlation_id=correlation_id,
                                       model=self.model, prompt_length=prompt_length)
                raise LLMProviderError(f"Anthropic API error: {e}")
