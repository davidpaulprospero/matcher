"""
Unified LLM client for matcher-pipeline.

This package provides a consistent interface for working with multiple LLM providers
(Gemini, Anthropic, Ollama) with built-in retry logic, caching, and JSON parsing.

Basic usage:
    from src.llm_client import create_client, LLMRequest, ResponseFormat

    # Create client
    client = create_client("gemini", api_key="your-api-key")

    # Make request
    request = LLMRequest(
        prompt="Extract keywords from: ...",
        response_format=ResponseFormat.JSON_ARRAY,
        cache_key_prefix="keywords"
    )

    # Get response
    response = client.generate(request)
    keywords = response.parsed_data  # Automatically parsed JSON

Features:
- Automatic retry with exponential backoff
- Unified caching with TTL support
- JSON parsing with multiple fallback strategies
- Consistent error handling
- Support for text and vision models
"""

from .base import LLMClient, LLMRequest, LLMResponse, ResponseFormat
from .factory import create_client, create_client_from_config
from .providers import GeminiClient, AnthropicClient, OllamaClient
from .exceptions import (
    LLMClientError,
    LLMTimeoutError,
    LLMParseError,
    LLMProviderError
)

__version__ = "1.0.0"

__all__ = [
    # Base classes
    "LLMClient",
    "LLMRequest",
    "LLMResponse",
    "ResponseFormat",

    # Factory functions
    "create_client",
    "create_client_from_config",

    # Provider implementations
    "GeminiClient",
    "AnthropicClient",
    "OllamaClient",

    # Exceptions
    "LLMClientError",
    "LLMTimeoutError",
    "LLMParseError",
    "LLMProviderError",
]
