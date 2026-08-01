"""LLM configuration: LLM providers, retry logic, caching.

Extracted from monolithic config.py during refactoring (Jan 7, 2026).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

__all__ = [
    'LLMRetryConfig',
    'LLMCacheConfig',
    'LLMProviderConfig',
    'LLMConfig',
]


@dataclass
class LLMRetryConfig:
    """Retry configuration for LLM calls

    Unified retry logic for all LLM providers with exponential backoff.
    """
    max_retries: int = 3
    retry_delay_seconds: float = 2.0
    timeout_seconds: int = 120
    exponential_backoff: bool = True


@dataclass
class LLMCacheConfig:
    """Cache configuration for LLM responses

    Unified caching with TTL support for all LLM operations.

    Quality tier tracking:
    - Cache entries are tagged with quality_tier based on confidence:
      - high: confidence >= 0.8
      - medium: 0.5 <= confidence < 0.8
      - low: confidence < 0.5
    - When expire_low_quality=True, low-quality cache entries are skipped,
      forcing a fresh LLM evaluation to potentially get better results.
    """
    enabled: bool = True
    ttl_hours: int = 24  # 0 = never expire
    cache_dir: str = ".cache/llm_responses"
    expire_low_quality: bool = False  # Skip cache entries with quality_tier='low'


@dataclass
class LLMProviderConfig:
    """Per-provider LLM settings

    Provider-specific configuration for models, tokens, temperature.
    """
    model: str
    max_tokens: int = 2000
    temperature: float = 0.7


@dataclass
class LLMConfig:
    """LLM settings for keyword extraction and other tasks

    Chain-of-thought: Centralizes LLM provider settings with unified client
    Reasoning: Multiple components need LLM access with consistent retry/caching
    Decision: Use src/llm_client/ package for all LLM operations

    New in v3.1: Unified LLM client with retry logic, caching, and JSON parsing.
    See CLAUDE.md Rule 9 for usage examples.
    """
    provider: str = "google"  # google, anthropic, minimax, ollama
    model: str = "gemini-2.5-flash"
    api_key: str = ""  # Loaded from environment if empty

    # Anthropic specific (legacy compatibility)
    anthropic_model: str = "claude-3-haiku-20240307"

    # Ollama specific
    ollama_model: str = "llama3.2"
    ollama_host: str = "http://localhost:11434"

    # Temperature and generation settings (legacy compatibility)
    temperature: float = 0.7
    max_tokens: int = 2000

    # Sub-configs for unified LLM client
    retry: LLMRetryConfig = field(default_factory=LLMRetryConfig)
    cache: LLMCacheConfig = field(default_factory=LLMCacheConfig)

    # Provider-specific configs
    gemini: LLMProviderConfig = field(default_factory=lambda: LLMProviderConfig(model="gemini-2.5-flash"))
    anthropic: LLMProviderConfig = field(default_factory=lambda: LLMProviderConfig(model="claude-3-haiku-20240307"))
    minimax: LLMProviderConfig = field(default_factory=lambda: LLMProviderConfig(model="MiniMax-M2.7"))
    ollama: LLMProviderConfig = field(default_factory=lambda: LLMProviderConfig(model="llama3.2"))

    def __post_init__(self):
        """Load API key from environment if not set and convert nested dicts"""

        # Load API key from environment
        if not self.api_key:
            if self.provider == "google":
                self.api_key = os.getenv("GEMINI_API_KEY", "")
            elif self.provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
            elif self.provider == "minimax":
                self.api_key = os.getenv("MINIMAX_API_KEY", "") or os.getenv("ANTHROPIC_API_KEY", "")

        # Convert nested dicts to dataclasses (Rule 2)
        if isinstance(self.retry, dict):
            self.retry = LLMRetryConfig(**self.retry)
        if isinstance(self.cache, dict):
            self.cache = LLMCacheConfig(**self.cache)
        if isinstance(self.gemini, dict):
            self.gemini = LLMProviderConfig(**self.gemini)
        if isinstance(self.anthropic, dict):
            self.anthropic = LLMProviderConfig(**self.anthropic)
        if isinstance(self.minimax, dict):
            self.minimax = LLMProviderConfig(**self.minimax)
        if isinstance(self.ollama, dict):
            self.ollama = LLMProviderConfig(**self.ollama)
