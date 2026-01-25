"""
Anthropic Claude LLM client implementation.
"""

import logging
from ..base import LLMClient, LLMRequest
from ..exceptions import LLMProviderError, LLMTimeoutError

logger = logging.getLogger(__name__)


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

            # Make API call
            response = self.client.messages.create(**params)

            # Extract text from response
            if response.content and len(response.content) > 0:
                return response.content[0].text
            else:
                logger.warning(f"Anthropic response has no content: {response}")
                return ""

        except Exception as e:
            error_msg = str(e).lower()

            # Check for timeout
            if "timeout" in error_msg or "timed out" in error_msg:
                raise LLMTimeoutError(f"Anthropic request timed out: {e}")

            # Check for common API errors
            if "api_key" in error_msg or "authentication" in error_msg or "unauthorized" in error_msg:
                raise LLMProviderError(f"Anthropic authentication error: {e}")
            elif "quota" in error_msg or "rate" in error_msg or "overloaded" in error_msg:
                raise LLMProviderError(f"Anthropic quota/rate limit error: {e}")
            else:
                raise LLMProviderError(f"Anthropic API error: {e}")
