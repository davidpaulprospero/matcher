"""
API Healer - LLM and external API error recovery.

Handles:
- Rate limiting (429 errors)
- Authentication failures (401/403)
- Quota exceeded
- Timeout errors
- Provider-specific errors (Gemini, Anthropic, Ollama)
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from ..base import Healer, HealerResult, HealerAction, HealerEvent, HealerEventData, get_config_value, set_config_value

if TYPE_CHECKING:
    from ...config import Config
    from ...state import PipelineState

logger = logging.getLogger(__name__)


class APIHealer(Healer):
    """
    Heals API-related errors with backoff and provider switching.

    Recovery strategies:
    1. Rate limit: Exponential backoff with retry
    2. Auth errors: Check API key, suggest fixes
    3. Quota exceeded: Switch provider or wait
    4. Timeout: Increase timeout, retry with smaller batch
    5. Provider errors: Switch to alternate provider
    """

    name = "api-healer"
    description = "Fix API rate limits and provider errors"

    error_patterns = [
        "rate limit",
        "429",
        "too many requests",
        "quota",
        "exceeded",
        "401",
        "403",
        "unauthorized",
        "forbidden",
        "api key",
        "timeout",
        "timed out",
        "connection",
        "gemini",
        "anthropic",
        "openai",
        "ollama",
    ]

    # Backoff configuration
    INITIAL_BACKOFF = 5.0  # seconds
    MAX_BACKOFF = 300.0    # 5 minutes max
    BACKOFF_MULTIPLIER = 2.0

    # Provider fallback order
    PROVIDER_FALLBACK = {
        "gemini": ["anthropic", "ollama"],
        "anthropic": ["gemini", "ollama"],
        "ollama": ["gemini", "anthropic"],
    }

    def __init__(self, config, project_dir):
        super().__init__(config, project_dir)
        self.backoff_time = self.INITIAL_BACKOFF
        self.retry_count = 0

    def fix(
        self,
        error: Exception,
        state: 'PipelineState',
        stage_name: str
    ) -> HealerResult:
        """Attempt to fix API-related errors."""
        error_str = str(error).lower()

        # Rate limiting - exponential backoff
        if any(p in error_str for p in ["rate limit", "429", "too many requests"]):
            return self._handle_rate_limit(error, state)

        # Authentication errors
        if any(p in error_str for p in ["401", "403", "unauthorized", "forbidden", "api key"]):
            return self._handle_auth_error(error, state)

        # Quota exceeded
        if any(p in error_str for p in ["quota", "exceeded", "billing"]):
            return self._handle_quota_exceeded(error, state)

        # Timeout errors
        if any(p in error_str for p in ["timeout", "timed out"]):
            return self._handle_timeout(error, state)

        # Connection errors
        if "connection" in error_str:
            return self._handle_connection_error(error, state)

        # Generic API error - try provider switch
        return self._try_provider_switch(error, state)

    def _handle_rate_limit(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Handle rate limiting with exponential backoff."""
        self.log_attempt(f"Rate limited, waiting {self.backoff_time:.1f}s...")

        time.sleep(self.backoff_time)

        # Increase backoff for next time
        old_backoff = self.backoff_time
        self.backoff_time = min(self.backoff_time * self.BACKOFF_MULTIPLIER, self.MAX_BACKOFF)
        self.retry_count += 1

        self.log_success(f"Waited {old_backoff:.1f}s, retrying (attempt {self.retry_count})")

        return HealerResult.fixed(
            f"Rate limit backoff: waited {old_backoff:.1f}s",
            action=HealerAction.RETRY,
            backoff_seconds=old_backoff,
            retry_count=self.retry_count
        )

    def _handle_auth_error(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Handle authentication errors."""
        self.log_attempt("Checking API authentication...")

        error_str = str(error)

        # Identify which provider failed
        provider = None
        if "gemini" in error_str.lower():
            provider = "gemini"
            env_var = "GEMINI_API_KEY"
        elif "anthropic" in error_str.lower():
            provider = "anthropic"
            env_var = "ANTHROPIC_API_KEY"
        elif "openai" in error_str.lower():
            provider = "openai"
            env_var = "OPENAI_API_KEY"
        else:
            env_var = "API_KEY"

        # Try switching to alternate provider
        if provider and provider in self.PROVIDER_FALLBACK:
            return self._try_provider_switch(error, state, exclude=provider)

        self.log_failure(f"Authentication failed - check {env_var} environment variable")
        return HealerResult.failed(
            f"API authentication failed. Verify {env_var} is set correctly.",
            provider=provider,
            env_var=env_var
        )

    def _handle_quota_exceeded(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Handle quota exceeded errors."""
        self.log_attempt("Quota exceeded, trying alternate provider...")

        # Try switching providers
        result = self._try_provider_switch(error, state)
        if result.success:
            return result

        # If no alternate, suggest waiting
        self.log_failure("Quota exceeded on all providers")
        return HealerResult.failed(
            "API quota exceeded. Try again later or add credits to your account."
        )

    def _handle_timeout(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Handle timeout errors."""
        self.log_attempt("Request timed out, adjusting settings...")

        # Try to increase timeout in config
        llm_config = getattr(self.config, 'llm', None)
        if llm_config:
            current_timeout = get_config_value(llm_config, 'timeout', 30)
            new_timeout = min(current_timeout * 2, 300)  # Max 5 minutes

            set_config_value(llm_config, 'timeout', new_timeout)

            self.log_success(f"Increased timeout: {current_timeout}s -> {new_timeout}s")
            return HealerResult.config_changed(
                f"Increased timeout from {current_timeout}s to {new_timeout}s",
                old_timeout=current_timeout,
                new_timeout=new_timeout
            )

        # Fallback: just retry with backoff
        time.sleep(5)
        return HealerResult.fixed("Timeout occurred, retrying after brief wait", action=HealerAction.RETRY)

    def _handle_connection_error(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Handle connection errors."""
        self.log_attempt("Connection error, retrying after brief wait...")

        time.sleep(self.INITIAL_BACKOFF)

        return HealerResult.fixed(
            "Connection error, retrying after brief wait",
            action=HealerAction.RETRY,
            waited_seconds=self.INITIAL_BACKOFF
        )

    def _try_provider_switch(
        self,
        error: Exception,
        state: 'PipelineState',
        exclude: str = None
    ) -> HealerResult:
        """Try switching to an alternate LLM provider."""
        self.log_attempt("Attempting provider switch...")

        llm_config = getattr(self.config, 'llm', None)
        if not llm_config:
            return HealerResult.failed("No LLM config available for provider switch")

        current_provider = get_config_value(llm_config, 'provider', None)
        if not current_provider:
            current_provider = get_config_value(llm_config, 'default_provider', 'gemini')

        # Get fallback list
        fallbacks = self.PROVIDER_FALLBACK.get(current_provider, [])
        if exclude:
            fallbacks = [p for p in fallbacks if p != exclude]

        for provider in fallbacks:
            # Check if provider is configured (has API key)
            if self._is_provider_available(provider):
                # Update config - try 'provider' first, then 'default_provider'
                if get_config_value(llm_config, 'provider') is not None:
                    set_config_value(llm_config, 'provider', provider)
                else:
                    set_config_value(llm_config, 'default_provider', provider)

                self.log_success(f"Switched provider: {current_provider} -> {provider}")
                return HealerResult.config_changed(
                    f"Switched LLM provider from {current_provider} to {provider}",
                    old_provider=current_provider,
                    new_provider=provider
                )

        return HealerResult.failed(f"No alternate providers available (tried: {fallbacks})")

    def _is_provider_available(self, provider: str) -> bool:
        """Check if a provider is available (has API key configured)."""
        import os

        key_mapping = {
            "gemini": "GEMINI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "ollama": None,  # Ollama doesn't need API key
        }

        env_var = key_mapping.get(provider)
        if env_var is None:  # Ollama
            return True  # Assume available

        return bool(os.environ.get(env_var))

    def handle_event(self, event_data: HealerEventData) -> None:
        """Handle cross-healer coordination events.

        Reacts to:
        - CONFIG_CHANGED: Reset backoff since config may have fixed the issue
        - RATE_LIMITED: Increase backoff preemptively
        - PROVIDER_SWITCHED: Reset backoff for the new provider
        """
        if event_data.event == HealerEvent.CONFIG_CHANGED:
            self.reset_backoff()
            logger.debug(f"[{self.name}] Reset backoff due to config change from {event_data.source_healer}")
        elif event_data.event == HealerEvent.RATE_LIMITED:
            # Another healer hit rate limits - increase our backoff preemptively
            self.backoff_time = min(self.backoff_time * self.BACKOFF_MULTIPLIER, self.MAX_BACKOFF)
            logger.debug(f"[{self.name}] Increased backoff to {self.backoff_time:.1f}s due to rate limit from {event_data.source_healer}")
        elif event_data.event == HealerEvent.PROVIDER_SWITCHED:
            # Provider changed - reset backoff for fresh start
            self.reset_backoff()
            logger.debug(f"[{self.name}] Reset backoff due to provider switch from {event_data.source_healer}")

    def reset_backoff(self):
        """Reset backoff state after successful operation."""
        self.backoff_time = self.INITIAL_BACKOFF
        self.retry_count = 0
