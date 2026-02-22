"""
Base classes and interfaces for LLM client abstraction.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from enum import Enum
import time
import logging
from typing import Optional as TypingOptional

logger = logging.getLogger(__name__)


def _get_correlation_id(request_correlation_id: TypingOptional[str] = None) -> TypingOptional[str]:
    """
    Get correlation ID from request or auto-fetch from pipeline context.

    Args:
        request_correlation_id: Correlation ID from request (if provided)

    Returns:
        Correlation ID string or None
    """
    if request_correlation_id:
        return request_correlation_id
    # Auto-fetch from pipeline context
    try:
        from src.logging_templates import get_correlation_id as pipeline_get_correlation_id
        return pipeline_get_correlation_id()
    except Exception:
        return None


def _format_correlation(correlation_id: TypingOptional[str]) -> str:
    """Format correlation ID into log prefix."""
    if correlation_id:
        return f"[corr:{correlation_id}]"
    return ""


class ResponseFormat(Enum):
    """Expected response format from LLM."""
    TEXT = "text"
    JSON = "json"
    JSON_ARRAY = "json_array"


@dataclass
class LLMRequest:
    """
    Unified request format for all LLM operations.

    Attributes:
        prompt: The main prompt text
        system_prompt: Optional system prompt (for providers that support it)
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature (0.0-1.0)
        response_format: Expected format of response
        timeout: Request timeout in seconds

        # Batch processing
        batch_prompts: Optional list of prompts for batch processing

        # Vision API support
        images: Optional list of image bytes for vision models
        image_format: Format of images (jpeg, png)

        # Caching
        cache_key_prefix: Prefix for cache key (e.g., "matching", "keywords")
        use_cache: Whether to use caching for this request

        # Correlation tracking
        correlation_id: Optional correlation ID for request tracing

        # Metadata
        metadata: Additional metadata for logging/tracking
    """
    prompt: str
    system_prompt: Optional[str] = None
    max_tokens: int = 2000
    temperature: float = 0.7
    response_format: ResponseFormat = ResponseFormat.TEXT
    timeout: int = 120

    # Batch processing
    batch_prompts: Optional[List[str]] = None

    # Vision API
    images: Optional[List[bytes]] = None
    image_format: str = "jpeg"

    # Caching
    cache_key_prefix: str = "default"
    use_cache: bool = True

    # Correlation tracking
    correlation_id: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """
    Unified response format from all LLM operations.

    Attributes:
        text: Raw text response from LLM
        parsed_data: Parsed JSON data (if response_format was JSON/JSON_ARRAY)
        confidence: Confidence score (0.0-1.0) if available
        provider: Provider name (gemini, anthropic, ollama)
        model: Model name used
        cached: Whether response was served from cache
        request_time_ms: Time taken for request in milliseconds
        tokens_used: Number of tokens used (if available)
        input_tokens: Number of input tokens (for cost calculation)
        output_tokens: Number of output tokens (for cost calculation)
        metadata: Additional response metadata
    """
    text: str
    parsed_data: Optional[Union[dict, List[dict]]] = None
    confidence: float = 1.0
    provider: str = ""
    model: str = ""
    cached: bool = False
    request_time_ms: float = 0.0
    tokens_used: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMClient(ABC):
    """
    Abstract base class for LLM providers.

    All provider implementations must extend this class and implement:
    - provider_name property
    - _call_api method

    The generate() method provides unified functionality:
    - Caching (if enabled)
    - Retry with exponential backoff
    - Response parsing based on format
    - Error handling
    """

    def __init__(self, api_key: str, model: str, cache_dir: str = ".cache/llm_responses", cache_ttl_hours: int = 24, cache_skip_low_quality: bool = False):
        """
        Initialize LLM client.

        Args:
            api_key: API key for the provider (empty for local models)
            model: Model name/ID
            cache_dir: Base directory for caching responses
            cache_ttl_hours: TTL for cached LLM responses in hours (0 = never expire)
            cache_skip_low_quality: If True, skip cache entries with quality_tier='low'
        """
        self.api_key = api_key
        self.model = model
        self.cache_dir = cache_dir
        self.cache_ttl_hours = cache_ttl_hours
        self.cache_skip_low_quality = cache_skip_low_quality
        self._cache = None  # Lazy initialization

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider name (e.g., 'gemini', 'anthropic', 'ollama')."""
        pass

    @abstractmethod
    def _call_api(self, request: LLMRequest) -> str:
        """
        Provider-specific API call implementation.

        This method should:
        - Make the actual API call to the provider
        - Return raw text response
        - Raise exceptions on errors

        It should NOT:
        - Handle retries (done by generate())
        - Parse JSON (done by generate())
        - Handle caching (done by generate())

        Args:
            request: LLM request object

        Returns:
            Raw text response from provider

        Raises:
            LLMProviderError: On provider-specific errors
            LLMTimeoutError: On timeout
        """
        pass

    @property
    def cache(self):
        """Lazy-load cache to avoid circular imports."""
        if self._cache is None:
            from .cache import LLMCache
            self._cache = LLMCache(
                self.cache_dir,
                provider=self.provider_name,
                ttl_hours=self.cache_ttl_hours,
                skip_low_quality=self.cache_skip_low_quality
            )
        return self._cache

    def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Main entry point for generating LLM responses.

        This method handles:
        1. Cache check (if enabled)
        2. API call with retry logic
        3. Response parsing based on format
        4. Cache storage
        5. Response wrapping

        Args:
            request: LLM request object

        Returns:
            LLM response object with text and parsed data

        Raises:
            LLMProviderError: On provider errors after retries
            LLMTimeoutError: On timeout
            LLMParseError: On JSON parsing failure (if strict)
        """
        start_time = time.time()

        # Get correlation ID for logging
        correlation_id = _get_correlation_id(request.correlation_id)
        corr = _format_correlation(correlation_id)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Estimate token count from prompt length (rough approximation: 1 token ≈ 4 chars)
        estimated_input_tokens = len(request.prompt) // 4
        estimated_total = estimated_input_tokens + request.max_tokens

        # Log request initiation at INFO level with token estimates
        logger.info(
            f"[{self.provider_name.upper()}] LLM request initiated{corr}: "
            f"model={self.model}, prompt_length={len(request.prompt)}, "
            f"estimated_input_tokens~{estimated_input_tokens}, "
            f"max_response_tokens={request.max_tokens}, estimated_total_tokens~{estimated_total}, "
            f"temperature={request.temperature}, response_format={request.response_format.value}, "
            f"timestamp={timestamp}"
        )

        # Check cache if enabled
        if request.use_cache:
            cached = self.cache.get(request)
            if cached:
                logger.debug(
                    f"[{self.provider_name.upper()}] Cache hit for request{corr}: "
                    f"model={self.model}, cache_key_prefix={request.cache_key_prefix}"
                )
                return LLMResponse(
                    text=cached['text'],
                    parsed_data=cached.get('parsed_data'),
                    provider=self.provider_name,
                    model=self.model,
                    cached=True,
                    request_time_ms=0.0
                )

        # Call API with retry logic (pass correlation ID to retry for logging)
        from .retry import with_retry
        text = with_retry(
            lambda: self._call_api(request),
            max_retries=3,
            timeout=request.timeout,
            correlation_id=correlation_id,
            provider=self.provider_name,
            model=self.model
        )

        # Extract token usage from provider (US-162-010)
        input_tokens = None
        output_tokens = None
        tokens_used = None

        # Check if provider has stored token usage
        if hasattr(self, '_last_tokens_used'):
            tokens_used = getattr(self, '_last_tokens_used', None)
            input_tokens = getattr(self, '_last_prompt_tokens', None)
            output_tokens = getattr(self, '_last_completion_tokens', None)

        # Parse response based on format
        parsed_data = None
        if request.response_format == ResponseFormat.JSON:
            from .parsers import parse_json
            parsed_data = parse_json(text)
        elif request.response_format == ResponseFormat.JSON_ARRAY:
            from .parsers import parse_json_array
            parsed_data = parse_json_array(text)

        # Calculate request time
        request_time_ms = (time.time() - start_time) * 1000

        # Create response with token info
        response = LLMResponse(
            text=text,
            parsed_data=parsed_data,
            provider=self.provider_name,
            model=self.model,
            cached=False,
            request_time_ms=request_time_ms,
            tokens_used=tokens_used,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

        # Cache result if enabled
        if request.use_cache:
            self.cache.set(request, response)

        # Log successful response at INFO level with response time and token count
        token_info = ""
        timing_breakdown = ""
        if response.input_tokens or response.output_tokens:
            total = response.tokens_used or 0
            input_tok = response.input_tokens or 0
            output_tok = response.output_tokens or 0
            token_info = f", input_tokens={input_tok}, output_tokens={output_tok}, total_tokens={total}"
            # Add timing breakdown at DEBUG level
            timing_breakdown = f", latency_ms={response.request_time_ms:.2f}"

        logger.info(
            f"[{self.provider_name.upper()}] LLM response successful{corr}: "
            f"model={self.model}, status=success{timing_breakdown}{token_info}"
        )

        # Log detailed timing and token info at DEBUG level
        logger.debug(
            f"[{self.provider_name.upper()}] LLM response details{corr}: "
            f"model={self.model}, latency_ms={response.request_time_ms:.2f}, "
            f"response_length={len(response.text)} chars, "
            f"tokens: input={response.input_tokens or 'N/A'}, "
            f"output={response.output_tokens or 'N/A'}, total={response.tokens_used or 'N/A'}, "
            f"cached={response.cached}"
        )

        return response
