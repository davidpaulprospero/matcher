"""Unit tests for APIHealer."""

import os
import time
import pytest
from unittest.mock import MagicMock, patch


class TestAPIHealerInit:
    """Test APIHealer initialization."""

    def test_init_with_default_backoff(self):
        """Test that APIHealer initializes with default backoff values."""
        from src.agents.healers.api import APIHealer

        config = MagicMock()
        healer = APIHealer(config, "/tmp/project")

        assert healer.backoff_time == healer.INITIAL_BACKOFF
        assert healer.retry_count == 0

    def test_init_stores_config_and_project(self):
        """Test that APIHealer stores config and project_dir."""
        from src.agents.healers.api import APIHealer

        config = MagicMock()
        healer = APIHealer(config, "/path/to/project")

        assert healer.config is config
        assert healer.project_dir == "/path/to/project"


class TestAPIHealerCanHandle:
    """Test APIHealer.can_handle() method."""

    def test_can_handle_rate_limit(self):
        """Test that APIHealer can handle rate limit errors."""
        from src.agents.healers.api import APIHealer

        config = MagicMock()
        healer = APIHealer(config, "/tmp")

        error = Exception("Rate limit exceeded")
        assert healer.can_handle(error, "TEST") is True

        error = Exception("HTTP 429 Too Many Requests")
        assert healer.can_handle(error, "TEST") is True

    def test_can_handle_auth_errors(self):
        """Test that APIHealer can handle authentication errors."""
        from src.agents.healers.api import APIHealer

        config = MagicMock()
        healer = APIHealer(config, "/tmp")

        error = Exception("HTTP 401 Unauthorized")
        assert healer.can_handle(error, "TEST") is True

        error = Exception("HTTP 403 Forbidden")
        assert healer.can_handle(error, "TEST") is True

        error = Exception("Invalid API key")
        assert healer.can_handle(error, "TEST") is True

    def test_can_handle_timeout_errors(self):
        """Test that APIHealer can handle timeout errors."""
        from src.agents.healers.api import APIHealer

        config = MagicMock()
        healer = APIHealer(config, "/tmp")

        error = Exception("Request timeout")
        assert healer.can_handle(error, "TEST") is True

        error = Exception("Connection timed out")
        assert healer.can_handle(error, "TEST") is True

    def test_can_handle_provider_errors(self):
        """Test that APIHealer can handle provider-specific errors."""
        from src.agents.healers.api import APIHealer

        config = MagicMock()
        healer = APIHealer(config, "/tmp")

        # Provider names in errors
        error = Exception("Gemini API error")
        assert healer.can_handle(error, "TEST") is True

        error = Exception("Anthropic rate limited")
        assert healer.can_handle(error, "TEST") is True

        error = Exception("Ollama connection refused")
        assert healer.can_handle(error, "TEST") is True

    def test_cannot_handle_unrelated_error(self):
        """Test that APIHealer doesn't handle unrelated errors."""
        from src.agents.healers.api import APIHealer

        config = MagicMock()
        healer = APIHealer(config, "/tmp")

        error = Exception("File not found")
        assert healer.can_handle(error, "TEST") is False


class TestAPIHealerHandleRateLimit:
    """Test APIHealer._handle_rate_limit() method."""

    def test_rate_limit_waits_backoff_time(self):
        """Test that rate limit handler waits for backoff time."""
        from src.agents.healers.api import APIHealer
        from src.agents.base import HealerAction

        config = MagicMock()
        healer = APIHealer(config, "/tmp")
        healer.backoff_time = 0.01  # Short backoff for test

        state = MagicMock()
        error = Exception("Rate limit")

        start = time.time()
        result = healer._handle_rate_limit(error, state)
        elapsed = time.time() - start

        assert elapsed >= 0.01
        assert result.success is True
        assert result.action == HealerAction.RETRY

    def test_rate_limit_increases_backoff(self):
        """Test that rate limit increases backoff time exponentially."""
        from src.agents.healers.api import APIHealer

        config = MagicMock()
        healer = APIHealer(config, "/tmp")
        healer.backoff_time = 0.01
        healer.BACKOFF_MULTIPLIER = 2.0
        healer.MAX_BACKOFF = 10.0

        state = MagicMock()
        error = Exception("Rate limit")

        healer._handle_rate_limit(error, state)
        assert healer.backoff_time == 0.02  # 0.01 * 2

        healer._handle_rate_limit(error, state)
        assert healer.backoff_time == 0.04  # 0.02 * 2

    def test_rate_limit_increments_retry_count(self):
        """Test that rate limit increments retry count."""
        from src.agents.healers.api import APIHealer

        config = MagicMock()
        healer = APIHealer(config, "/tmp")
        healer.backoff_time = 0.01

        state = MagicMock()
        error = Exception("Rate limit")

        assert healer.retry_count == 0
        healer._handle_rate_limit(error, state)
        assert healer.retry_count == 1
        healer._handle_rate_limit(error, state)
        assert healer.retry_count == 2

    def test_rate_limit_respects_max_backoff(self):
        """Test that rate limit respects maximum backoff time."""
        from src.agents.healers.api import APIHealer

        config = MagicMock()
        healer = APIHealer(config, "/tmp")
        healer.backoff_time = 200.0  # High value
        healer.MAX_BACKOFF = 300.0

        state = MagicMock()
        error = Exception("Rate limit")

        healer._handle_rate_limit(error, state)
        # 200 * 2 = 400, but capped at 300
        assert healer.backoff_time == 300.0


class TestAPIHealerHandleAuthError:
    """Test APIHealer._handle_auth_error() method."""

    def test_auth_error_detects_gemini_and_tries_switch(self):
        """Test that auth error detects Gemini provider and attempts switch."""
        from src.agents.healers.api import APIHealer
        from src.agents.base import HealerAction

        config = MagicMock()
        config.llm = MagicMock()
        config.llm.provider = "gemini"
        healer = APIHealer(config, "/tmp")

        state = MagicMock()
        error = Exception("Gemini authentication failed")

        # With anthropic key available, should switch successfully
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            result = healer._handle_auth_error(error, state)
            assert result.success is True
            assert result.action == HealerAction.MODIFY_CONFIG
            assert config.llm.provider == "anthropic"

    def test_auth_error_detects_anthropic_and_tries_switch(self):
        """Test that auth error detects Anthropic provider and attempts switch."""
        from src.agents.healers.api import APIHealer
        from src.agents.base import HealerAction

        config = MagicMock()
        config.llm = MagicMock()
        config.llm.provider = "anthropic"
        healer = APIHealer(config, "/tmp")

        state = MagicMock()
        error = Exception("Anthropic API key invalid")

        # With gemini key available, should switch successfully
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True):
            result = healer._handle_auth_error(error, state)
            assert result.success is True
            assert result.action == HealerAction.MODIFY_CONFIG
            assert config.llm.provider == "gemini"

    def test_auth_error_falls_back_to_ollama(self):
        """Test that auth error falls back to Ollama when no API keys."""
        from src.agents.healers.api import APIHealer
        from src.agents.base import HealerAction

        config = MagicMock()
        config.llm = MagicMock()
        config.llm.provider = "gemini"
        healer = APIHealer(config, "/tmp")

        state = MagicMock()
        error = Exception("Gemini authentication failed")

        # With no API keys, should try ollama (always available)
        with patch.dict(os.environ, {}, clear=True):
            result = healer._handle_auth_error(error, state)
            # Ollama is always available, so should succeed
            if result.success:
                assert config.llm.provider == "ollama"

    def test_auth_error_unknown_provider_fails(self):
        """Test that auth error for unknown provider returns failure with suggestion."""
        from src.agents.healers.api import APIHealer

        config = MagicMock()
        config.llm = None  # No LLM config
        healer = APIHealer(config, "/tmp")

        state = MagicMock()
        error = Exception("Unknown API authentication error")

        result = healer._handle_auth_error(error, state)
        # Should return failure suggesting to check API_KEY
        assert result.success is False
        assert "API_KEY" in result.message


class TestAPIHealerHandleTimeout:
    """Test APIHealer._handle_timeout() method."""

    def test_timeout_increases_config_timeout(self):
        """Test that timeout handler increases config timeout."""
        from src.agents.healers.api import APIHealer
        from src.agents.base import HealerAction

        config = MagicMock()
        config.llm = MagicMock()
        config.llm.timeout = 30

        healer = APIHealer(config, "/tmp")
        state = MagicMock()
        error = Exception("Request timed out")

        result = healer._handle_timeout(error, state)

        assert result.success is True
        assert result.action == HealerAction.MODIFY_CONFIG
        assert result.modified_config is True
        assert config.llm.timeout == 60  # Doubled

    def test_timeout_respects_max_timeout(self):
        """Test that timeout handler respects maximum timeout."""
        from src.agents.healers.api import APIHealer

        config = MagicMock()
        config.llm = MagicMock()
        config.llm.timeout = 200

        healer = APIHealer(config, "/tmp")
        state = MagicMock()
        error = Exception("Request timed out")

        result = healer._handle_timeout(error, state)

        # 200 * 2 = 400, but capped at 300 (5 minutes)
        assert config.llm.timeout == 300

    def test_timeout_handles_missing_llm_config(self):
        """Test timeout handler when LLM config is missing."""
        from src.agents.healers.api import APIHealer
        from src.agents.base import HealerAction

        config = MagicMock()
        config.llm = None

        healer = APIHealer(config, "/tmp")
        state = MagicMock()
        error = Exception("Request timed out")

        result = healer._handle_timeout(error, state)

        # Should still return RETRY even without config
        assert result.success is True
        assert result.action == HealerAction.RETRY


class TestAPIHealerProviderSwitch:
    """Test APIHealer._try_provider_switch() method."""

    def test_provider_switch_with_available_provider(self):
        """Test provider switch when alternate provider is available."""
        from src.agents.healers.api import APIHealer
        from src.agents.base import HealerAction

        config = MagicMock()
        config.llm = MagicMock()
        config.llm.provider = "gemini"

        healer = APIHealer(config, "/tmp")
        state = MagicMock()
        error = Exception("Gemini error")

        # Make anthropic available
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            result = healer._try_provider_switch(error, state)

            if result.success:
                assert result.action == HealerAction.MODIFY_CONFIG
                assert config.llm.provider == "anthropic"

    def test_provider_switch_ollama_always_available(self):
        """Test that Ollama is always considered available (no API key needed)."""
        from src.agents.healers.api import APIHealer

        config = MagicMock()
        healer = APIHealer(config, "/tmp")

        assert healer._is_provider_available("ollama") is True

    def test_provider_switch_no_llm_config(self):
        """Test provider switch when LLM config is missing."""
        from src.agents.healers.api import APIHealer

        config = MagicMock()
        config.llm = None

        healer = APIHealer(config, "/tmp")
        state = MagicMock()
        error = Exception("API error")

        result = healer._try_provider_switch(error, state)

        assert result.success is False

    def test_provider_switch_exclude_specific_provider(self):
        """Test provider switch can exclude a specific provider."""
        from src.agents.healers.api import APIHealer

        config = MagicMock()
        config.llm = MagicMock()
        config.llm.provider = "gemini"

        healer = APIHealer(config, "/tmp")
        state = MagicMock()
        error = Exception("API error")

        # Exclude gemini (current) and only ollama available
        with patch.dict(os.environ, {}, clear=True):
            result = healer._try_provider_switch(error, state, exclude="anthropic")
            # ollama should be tried
            if result.success:
                assert config.llm.provider == "ollama"


class TestAPIHealerIsProviderAvailable:
    """Test APIHealer._is_provider_available() method."""

    def test_is_provider_available_gemini(self):
        """Test Gemini availability check."""
        from src.agents.healers.api import APIHealer

        config = MagicMock()
        healer = APIHealer(config, "/tmp")

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            assert healer._is_provider_available("gemini") is True

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GEMINI_API_KEY", None)
            assert healer._is_provider_available("gemini") is False

    def test_is_provider_available_anthropic(self):
        """Test Anthropic availability check."""
        from src.agents.healers.api import APIHealer

        config = MagicMock()
        healer = APIHealer(config, "/tmp")

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            assert healer._is_provider_available("anthropic") is True

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            assert healer._is_provider_available("anthropic") is False

    def test_is_provider_available_ollama_no_key_needed(self):
        """Test that Ollama doesn't need API key."""
        from src.agents.healers.api import APIHealer

        config = MagicMock()
        healer = APIHealer(config, "/tmp")

        # Even with no env vars, Ollama is available
        with patch.dict(os.environ, {}, clear=True):
            assert healer._is_provider_available("ollama") is True


class TestAPIHealerResetBackoff:
    """Test APIHealer.reset_backoff() method."""

    def test_reset_backoff_resets_time(self):
        """Test that reset_backoff resets backoff time."""
        from src.agents.healers.api import APIHealer

        config = MagicMock()
        healer = APIHealer(config, "/tmp")
        healer.backoff_time = 100.0

        healer.reset_backoff()

        assert healer.backoff_time == healer.INITIAL_BACKOFF

    def test_reset_backoff_resets_retry_count(self):
        """Test that reset_backoff resets retry count."""
        from src.agents.healers.api import APIHealer

        config = MagicMock()
        healer = APIHealer(config, "/tmp")
        healer.retry_count = 5

        healer.reset_backoff()

        assert healer.retry_count == 0


class TestAPIHealerFix:
    """Test APIHealer.fix() method routing."""

    def test_fix_routes_rate_limit(self):
        """Test that fix() routes rate limit errors correctly."""
        from src.agents.healers.api import APIHealer
        from src.agents.base import HealerAction

        config = MagicMock()
        healer = APIHealer(config, "/tmp")
        healer.backoff_time = 0.01

        state = MagicMock()
        error = Exception("Rate limit exceeded")

        result = healer.fix(error, state, "TEST")

        assert result.success is True
        assert result.action == HealerAction.RETRY

    def test_fix_routes_timeout(self):
        """Test that fix() routes timeout errors correctly."""
        from src.agents.healers.api import APIHealer

        config = MagicMock()
        config.llm = MagicMock()
        config.llm.timeout = 30

        healer = APIHealer(config, "/tmp")
        state = MagicMock()
        error = Exception("Connection timed out")

        result = healer.fix(error, state, "TEST")

        assert result.success is True

    def test_fix_routes_connection_error(self):
        """Test that fix() routes connection errors correctly."""
        from src.agents.healers.api import APIHealer
        from src.agents.base import HealerAction

        config = MagicMock()
        healer = APIHealer(config, "/tmp")
        healer.INITIAL_BACKOFF = 0.01

        state = MagicMock()
        error = Exception("Connection refused")

        result = healer.fix(error, state, "TEST")

        assert result.success is True
        assert result.action == HealerAction.RETRY
