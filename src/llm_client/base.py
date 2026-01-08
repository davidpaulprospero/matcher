"""
Base classes and interfaces for LLM client abstraction.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from enum import Enum
import time


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

    def __init__(self, api_key: str, model: str, cache_dir: str = ".cache/llm_responses"):
        """
        Initialize LLM client.

        Args:
            api_key: API key for the provider (empty for local models)
            model: Model name/ID
            cache_dir: Base directory for caching responses
        """
        self.api_key = api_key
        self.model = model
        self.cache_dir = cache_dir
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
            self._cache = LLMCache(self.cache_dir, provider=self.provider_name)
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

        # Check cache if enabled
        if request.use_cache:
            cached = self.cache.get(request)
            if cached:
                return LLMResponse(
                    text=cached['text'],
                    parsed_data=cached.get('parsed_data'),
                    provider=self.provider_name,
                    model=self.model,
                    cached=True,
                    request_time_ms=0.0
                )

        # Call API with retry logic
        from .retry import with_retry
        text = with_retry(
            lambda: self._call_api(request),
            max_retries=3,
            timeout=request.timeout
        )

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

        # Create response
        response = LLMResponse(
            text=text,
            parsed_data=parsed_data,
            provider=self.provider_name,
            model=self.model,
            cached=False,
            request_time_ms=request_time_ms
        )

        # Cache result if enabled
        if request.use_cache:
            self.cache.set(request, response)

        return response
