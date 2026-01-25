"""
Factory for creating LLM clients.
"""

import os
import logging
from typing import Optional
from .base import LLMClient
from .providers import GeminiClient, AnthropicClient, OllamaClient
from .exceptions import LLMProviderError

logger = logging.getLogger(__name__)


def create_client(
    provider: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    cache_dir: str = ".cache/llm_responses",
    cache_ttl_hours: int = 24,
    **kwargs
) -> LLMClient:
    """
    Factory function to create an LLM client.

    Args:
        provider: Provider name ("gemini", "anthropic", "ollama", or "google")
        api_key: API key (optional, will check env vars if not provided)
        model: Model name (optional, uses provider default if not provided)
        cache_dir: Cache directory
        cache_ttl_hours: TTL for cached LLM responses in hours (0 = never expire)
        **kwargs: Additional provider-specific arguments (e.g., host for Ollama)

    Returns:
        Configured LLM client instance

    Raises:
        LLMProviderError: If provider is invalid or configuration is incomplete

    Examples:
        # Gemini with explicit API key
        client = create_client("gemini", api_key="...", model="gemini-2.0-flash")

        # Anthropic with env var API key
        client = create_client("anthropic")  # Uses ANTHROPIC_API_KEY env var

        # Ollama (no API key needed)
        client = create_client("ollama", model="llama3.2", host="http://localhost:11434")
    """
    provider = provider.lower()

    # Normalize provider names
    if provider in ("google", "gemini"):
        provider = "gemini"
    elif provider in ("claude", "anthropic"):
        provider = "anthropic"

    # Create client based on provider
    if provider == "gemini":
        # Resolve API key
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise LLMProviderError(
                "Gemini API key required. Provide via api_key parameter or "
                "GEMINI_API_KEY / GOOGLE_API_KEY environment variable."
            )

        # Use default model if not provided
        if not model:
            model = "gemini-2.0-flash"

        return GeminiClient(
            api_key=api_key,
            model=model,
            cache_dir=cache_dir,
            cache_ttl_hours=cache_ttl_hours
        )

    elif provider == "anthropic":
        # Resolve API key
        if not api_key:
            api_key = os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            raise LLMProviderError(
                "Anthropic API key required. Provide via api_key parameter or "
                "ANTHROPIC_API_KEY environment variable."
            )

        # Use default model if not provided
        if not model:
            model = "claude-3-haiku-20240307"

        return AnthropicClient(
            api_key=api_key,
            model=model,
            cache_dir=cache_dir,
            cache_ttl_hours=cache_ttl_hours
        )

    elif provider == "ollama":
        # Ollama doesn't need API key
        if not model:
            model = "llama3.2"

        # Get host from kwargs or use default
        host = kwargs.get("host", "http://localhost:11434")

        return OllamaClient(
            model=model,
            host=host,
            cache_dir=cache_dir,
            cache_ttl_hours=cache_ttl_hours
        )

    else:
        raise LLMProviderError(
            f"Unknown provider '{provider}'. "
            f"Supported providers: gemini, anthropic, ollama"
        )


def create_client_from_config(config: "Config", cache_dir: Optional[str] = None) -> LLMClient:
    """
    Create LLM client from Config object.

    This is a convenience function for creating a client from the application's
    config system.

    Args:
        config: Config object with llm section
        cache_dir: Override cache directory (uses config.cache_dir if not provided)

    Returns:
        Configured LLM client

    Example:
        from src.config import load_config
        config = load_config("config.yaml")
        client = create_client_from_config(config)
    """
    llm_config = config.llm

    # Determine cache directory
    if not cache_dir:
        cache_dir = getattr(config, 'cache_dir', '.cache')
        cache_dir = f"{cache_dir}/llm_responses"

    # Determine provider
    provider = getattr(llm_config, 'provider', 'google')

    # Determine API key
    api_key = getattr(llm_config, 'api_key', None)

    # Determine model
    model = getattr(llm_config, 'model', None)

    # Determine cache TTL (from config.llm.cache.ttl_hours)
    cache_config = getattr(llm_config, 'cache', None)
    if cache_config:
        cache_ttl_hours = getattr(cache_config, 'ttl_hours', 24)
    else:
        cache_ttl_hours = 24

    # Create client
    return create_client(
        provider=provider,
        api_key=api_key,
        model=model,
        cache_dir=cache_dir,
        cache_ttl_hours=cache_ttl_hours
    )
