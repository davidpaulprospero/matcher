"""
Auto-routing LLM client helper.

Picks the right provider based on availability:
1. Ollama (preferred during Gemini→Ollama migration)
2. Gemini (if GEMINI_API_KEY / GOOGLE_API_KEY set)
3. Anthropic (if ANTHROPIC_API_KEY set)

This was added during the 2026-08-01 Gemini→Ollama migration to centralize
the provider-selection logic instead of every call site hardcoding Gemini.
"""

import logging
import os
from typing import Optional

from .base import LLMClient
from .factory import create_client

logger = logging.getLogger(__name__)


def create_auto_client(
    config: Optional[object] = None,
    *,
    preferred_provider: Optional[str] = None,
    preferred_model: Optional[str] = None,
    cache_dir: str = ".cache/llm_responses",
) -> LLMClient:
    """
    Create an LLM client using the best available provider.

    Order of preference (unless preferred_provider overrides):
    1. Ollama (local, free)
    2. Gemini (cloud, requires API key)
    3. Anthropic (cloud, requires API key)

    Args:
        config: Optional config object (reads .llm.* and .embedding.* sections)
        preferred_provider: Force a specific provider ('ollama', 'gemini', 'anthropic')
        preferred_model: Override the default model for the chosen provider
        cache_dir: Cache directory for LLM responses

    Returns:
        LLMClient instance

    Raises:
        RuntimeError: If no provider is available
    """
    # 1. Determine desired provider
    provider = preferred_provider

    if provider is None and config is not None:
        # Read from config.llm.provider
        if hasattr(config, 'llm') and hasattr(config.llm, 'provider'):
            provider = config.llm.provider

    # 2. Default to ollama
    if provider is None:
        provider = 'ollama'

    provider = provider.lower()

    # 3. Try the requested provider first
    try:
        if provider == 'ollama':
            host = 'http://localhost:11434'
            if config is not None and hasattr(config, 'llm') and hasattr(config.llm, 'ollama_host'):
                host = config.llm.ollama_host
            model = preferred_model or _get_model(config, 'ollama', 'gemma3:4b')
            logger.debug(f"Auto-routing to Ollama (model={model}, host={host})")
            return create_client('ollama', model=model, host=host, cache_dir=cache_dir)

        if provider == 'gemini':
            api_key = _get_gemini_key(config)
            if not api_key:
                raise RuntimeError("No GEMINI_API_KEY available")
            model = preferred_model or _get_model(config, 'gemini', 'gemini-2.5-flash')
            return create_client('gemini', api_key=api_key, model=model, cache_dir=cache_dir)

        if provider == 'anthropic':
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise RuntimeError("No ANTHROPIC_API_KEY available")
            model = preferred_model or _get_model(config, 'anthropic', 'claude-3-haiku-20240307')
            return create_client('anthropic', api_key=api_key, model=model, cache_dir=cache_dir)

    except Exception as e:
        logger.debug(f"Requested provider '{provider}' unavailable: {e}")

    # 4. Fallback: try Ollama, then Gemini, then Anthropic
    for fallback in ('ollama', 'gemini', 'anthropic'):
        if fallback == provider:
            continue  # already tried
        try:
            if fallback == 'ollama':
                host = 'http://localhost:11434'
                if config is not None and hasattr(config, 'llm') and hasattr(config.llm, 'ollama_host'):
                    host = config.llm.ollama_host
                model = preferred_model or _get_model(config, 'ollama', 'gemma3:4b')
                logger.info(f"Falling back to Ollama (model={model}, host={host})")
                return create_client('ollama', model=model, host=host, cache_dir=cache_dir)

            if fallback == 'gemini':
                api_key = _get_gemini_key(config)
                if not api_key:
                    continue
                model = preferred_model or _get_model(config, 'gemini', 'gemini-2.5-flash')
                logger.info(f"Falling back to Gemini (model={model})")
                return create_client('gemini', api_key=api_key, model=model, cache_dir=cache_dir)

            if fallback == 'anthropic':
                api_key = os.getenv('ANTHROPIC_API_KEY')
                if not api_key:
                    continue
                model = preferred_model or _get_model(config, 'anthropic', 'claude-3-haiku-20240307')
                logger.info(f"Falling back to Anthropic (model={model})")
                return create_client('anthropic', api_key=api_key, model=model, cache_dir=cache_dir)

        except Exception as e:
            logger.debug(f"Fallback provider '{fallback}' failed: {e}")
            continue

    raise RuntimeError(
        "No LLM provider available. Start Ollama (ollama serve) or set "
        "GEMINI_API_KEY / ANTHROPIC_API_KEY environment variable."
    )


def _get_gemini_key(config: Optional[object]) -> Optional[str]:
    """Get Gemini API key from config or environment."""
    if config is not None and hasattr(config, 'gemini_api_key'):
        key = config.gemini_api_key
        if key:
            return key
    return os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')


def _get_model(config: Optional[object], provider: str, default: str) -> str:
    """Get the model name for a provider from config."""
    if config is None or not hasattr(config, 'llm'):
        return default

    llm = config.llm

    # Provider-specific fields
    if provider == 'ollama' and hasattr(llm, 'ollama_model'):
        return llm.ollama_model
    if provider == 'gemini' and hasattr(llm, 'gemini_model'):
        return llm.gemini_model
    if provider == 'anthropic' and hasattr(llm, 'anthropic_model'):
        return llm.anthropic_model

    # Generic model field
    if hasattr(llm, 'model'):
        return llm.model

    return default
