"""
Unit tests for LLM client factory.
"""

import pytest
import os
from src.llm_client.factory import create_client
from src.llm_client.providers import GeminiClient, AnthropicClient, OllamaClient
from src.llm_client.exceptions import LLMProviderError


class TestCreateClient:
    """Test create_client factory function."""

    @pytest.mark.fast
    def test_create_gemini_client(self):
        """Test creating Gemini client."""
        client = create_client(
            provider="gemini",
            api_key="test_key",
            model="gemini-2.0-flash"
        )

        assert isinstance(client, GeminiClient)
        assert client.api_key == "test_key"
        assert client.model == "gemini-2.0-flash"
        assert client.provider_name == "gemini"

    @pytest.mark.fast
    def test_create_gemini_with_google_alias(self):
        """Test creating Gemini client using 'google' alias."""
        client = create_client(
            provider="google",
            api_key="test_key"
        )

        assert isinstance(client, GeminiClient)
        assert client.provider_name == "gemini"

    @pytest.mark.fast
    def test_create_anthropic_client(self):
        """Test creating Anthropic client."""
        client = create_client(
            provider="anthropic",
            api_key="test_key",
            model="claude-3-haiku-20240307"
        )

        assert isinstance(client, AnthropicClient)
        assert client.api_key == "test_key"
        assert client.model == "claude-3-haiku-20240307"
        assert client.provider_name == "anthropic"

    @pytest.mark.fast
    def test_create_anthropic_with_claude_alias(self):
        """Test creating Anthropic client using 'claude' alias."""
        client = create_client(
            provider="claude",
            api_key="test_key"
        )

        assert isinstance(client, AnthropicClient)
        assert client.provider_name == "anthropic"

    @pytest.mark.fast
    def test_create_ollama_client(self):
        """Test creating Ollama client."""
        client = create_client(
            provider="ollama",
            model="llama3.2",
            host="http://localhost:11434"
        )

        assert isinstance(client, OllamaClient)
        assert client.model == "llama3.2"
        assert client.host == "http://localhost:11434"
        assert client.provider_name == "ollama"

    @pytest.mark.fast
    def test_default_models(self):
        """Test default models are used when not specified."""
        # Gemini default
        client = create_client(provider="gemini", api_key="test")
        assert client.model == "gemini-2.0-flash"

        # Anthropic default
        client = create_client(provider="anthropic", api_key="test")
        assert client.model == "claude-3-haiku-20240307"

        # Ollama default
        client = create_client(provider="ollama")
        assert client.model == "llama3.2"

    @pytest.mark.fast
    def test_custom_cache_dir(self):
        """Test custom cache directory."""
        client = create_client(
            provider="gemini",
            api_key="test",
            cache_dir=".custom_cache"
        )

        assert client.cache_dir == ".custom_cache"

    @pytest.mark.requires_api
    def test_gemini_missing_api_key_raises_error(self, monkeypatch):
        """Test that missing Gemini API key raises error."""
        # Clear environment variables
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        with pytest.raises(LLMProviderError) as exc_info:
            create_client(provider="gemini")

        assert "API key required" in str(exc_info.value)

    @pytest.mark.requires_api
    def test_anthropic_missing_api_key_raises_error(self, monkeypatch):
        """Test that missing Anthropic API key raises error."""
        # Clear environment variable
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with pytest.raises(LLMProviderError) as exc_info:
            create_client(provider="anthropic")

        assert "API key required" in str(exc_info.value)

    @pytest.mark.fast
    def test_ollama_no_api_key_required(self):
        """Test that Ollama doesn't require API key."""
        # Should not raise error
        client = create_client(provider="ollama")
        assert isinstance(client, OllamaClient)

    @pytest.mark.requires_api
    def test_gemini_api_key_from_env(self, monkeypatch):
        """Test Gemini API key loaded from environment."""
        monkeypatch.setenv("GEMINI_API_KEY", "env_key")

        client = create_client(provider="gemini")

        assert client.api_key == "env_key"

    @pytest.mark.requires_api
    def test_gemini_api_key_from_google_api_key_env(self, monkeypatch):
        """Test Gemini API key loaded from GOOGLE_API_KEY."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "google_key")

        client = create_client(provider="gemini")

        assert client.api_key == "google_key"

    @pytest.mark.requires_api
    def test_anthropic_api_key_from_env(self, monkeypatch):
        """Test Anthropic API key loaded from environment."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env_key")

        client = create_client(provider="anthropic")

        assert client.api_key == "env_key"

    @pytest.mark.requires_api
    def test_explicit_api_key_overrides_env(self, monkeypatch):
        """Test explicit API key overrides environment variable."""
        monkeypatch.setenv("GEMINI_API_KEY", "env_key")

        client = create_client(
            provider="gemini",
            api_key="explicit_key"
        )

        assert client.api_key == "explicit_key"

    @pytest.mark.fast
    def test_unknown_provider_raises_error(self):
        """Test that unknown provider raises error."""
        with pytest.raises(LLMProviderError) as exc_info:
            create_client(provider="unknown_provider", api_key="test")

        assert "Unknown provider" in str(exc_info.value)
        assert "gemini" in str(exc_info.value).lower()
        assert "anthropic" in str(exc_info.value).lower()
        assert "ollama" in str(exc_info.value).lower()

    @pytest.mark.fast
    def test_case_insensitive_provider(self):
        """Test provider name is case-insensitive."""
        client1 = create_client(provider="GEMINI", api_key="test")
        client2 = create_client(provider="Gemini", api_key="test")
        client3 = create_client(provider="gemini", api_key="test")

        assert all(isinstance(c, GeminiClient) for c in [client1, client2, client3])

    @pytest.mark.fast
    def test_ollama_custom_host(self):
        """Test Ollama with custom host."""
        client = create_client(
            provider="ollama",
            model="llama3.2",
            host="http://192.168.1.100:11434"
        )

        assert client.host == "http://192.168.1.100:11434"

    @pytest.mark.fast
    def test_kwargs_passed_to_provider(self):
        """Test that additional kwargs are passed to provider."""
        client = create_client(
            provider="ollama",
            model="mistral",
            host="http://custom:8080"
        )

        assert client.model == "mistral"
        assert client.host == "http://custom:8080"

    @pytest.mark.fast
    def test_multiple_clients_independent(self):
        """Test creating multiple clients doesn't interfere."""
        client1 = create_client(provider="gemini", api_key="key1", model="model1")
        client2 = create_client(provider="gemini", api_key="key2", model="model2")

        assert client1.api_key == "key1"
        assert client2.api_key == "key2"
        assert client1.model == "model1"
        assert client2.model == "model2"


class TestCreateClientFromConfig:
    """Test create_client_from_config function."""

    @pytest.mark.fast
    def test_basic_usage(self):
        """Test creating client from config object."""
        from dataclasses import dataclass
        from src.llm_client.factory import create_client_from_config

        @dataclass
        class MockLLMConfig:
            provider: str = "gemini"
            model: str = "gemini-2.0-flash"
            api_key: str = "test_key"

        @dataclass
        class MockConfig:
            llm: MockLLMConfig = None
            cache_dir: str = ".cache"

        config = MockConfig()
        config.llm = MockLLMConfig()

        client = create_client_from_config(config)

        assert isinstance(client, GeminiClient)
        assert client.api_key == "test_key"
        assert client.model == "gemini-2.0-flash"
