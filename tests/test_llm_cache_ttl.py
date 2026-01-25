"""Tests for LLM cache TTL configuration (US-012).

Tests that the llm.cache.ttl_hours config setting is respected throughout the
LLM client stack, including:
- LLMCacheConfig dataclass field
- LLMCache class TTL handling
- LLMClient passing TTL to cache
- Factory functions using config TTL
"""

import json
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

pytestmark = pytest.mark.unit


# =============================================================================
# Test LLMCacheConfig Field
# =============================================================================

class TestLLMCacheConfigField:
    """Tests for llm.cache.ttl_hours field in LLMCacheConfig."""

    def test_llm_cache_config_has_ttl_hours_field(self):
        """LLMCacheConfig dataclass has ttl_hours field."""
        from src.config.sections.llm import LLMCacheConfig
        config = LLMCacheConfig()
        assert hasattr(config, 'ttl_hours')

    def test_llm_cache_config_ttl_hours_default_is_24(self):
        """Default ttl_hours is 24 (hours)."""
        from src.config.sections.llm import LLMCacheConfig
        config = LLMCacheConfig()
        assert config.ttl_hours == 24

    def test_llm_cache_config_ttl_hours_can_be_set_to_1(self):
        """ttl_hours can be set to 1 hour."""
        from src.config.sections.llm import LLMCacheConfig
        config = LLMCacheConfig(ttl_hours=1)
        assert config.ttl_hours == 1

    def test_llm_cache_config_ttl_hours_can_be_set_to_0(self):
        """ttl_hours=0 means never expire."""
        from src.config.sections.llm import LLMCacheConfig
        config = LLMCacheConfig(ttl_hours=0)
        assert config.ttl_hours == 0

    def test_llm_cache_config_ttl_hours_type_is_int(self):
        """ttl_hours field type is int."""
        from src.config.sections.llm import LLMCacheConfig
        from dataclasses import fields
        ttl_field = next(f for f in fields(LLMCacheConfig) if f.name == 'ttl_hours')
        # With `from __future__ import annotations`, type is stored as string
        assert ttl_field.type in (int, 'int')


# =============================================================================
# Test LLMConfig Cache Section
# =============================================================================

class TestLLMConfigCacheSection:
    """Tests for llm.cache nested config in LLMConfig."""

    def test_llm_config_has_cache_section(self):
        """LLMConfig has cache field (LLMCacheConfig)."""
        from src.config.sections.llm import LLMConfig
        config = LLMConfig()
        assert hasattr(config, 'cache')

    def test_llm_config_cache_has_ttl_hours(self):
        """LLMConfig.cache has ttl_hours field."""
        from src.config.sections.llm import LLMConfig
        config = LLMConfig()
        assert hasattr(config.cache, 'ttl_hours')

    def test_llm_config_cache_ttl_default_24(self):
        """LLMConfig.cache.ttl_hours defaults to 24."""
        from src.config.sections.llm import LLMConfig
        config = LLMConfig()
        assert config.cache.ttl_hours == 24


# =============================================================================
# Test LLMCache TTL Handling
# =============================================================================

class TestLLMCacheTTLHandling:
    """Tests for LLMCache class TTL handling."""

    def test_llm_cache_accepts_ttl_hours_parameter(self, tmp_path):
        """LLMCache.__init__ accepts ttl_hours parameter."""
        from src.llm_client.cache import LLMCache
        cache = LLMCache(base_dir=str(tmp_path), provider="test", ttl_hours=48)
        assert cache.ttl_seconds == 48 * 3600

    def test_llm_cache_ttl_hours_default_is_24(self, tmp_path):
        """LLMCache ttl_hours defaults to 24."""
        from src.llm_client.cache import LLMCache
        cache = LLMCache(base_dir=str(tmp_path), provider="test")
        assert cache.ttl_seconds == 24 * 3600

    def test_llm_cache_ttl_hours_0_disables_expiration(self, tmp_path):
        """LLMCache with ttl_hours=0 never expires entries."""
        from src.llm_client.cache import LLMCache
        cache = LLMCache(base_dir=str(tmp_path), provider="test", ttl_hours=0)
        assert cache.ttl_seconds == 0

    def test_llm_cache_ttl_hours_1_expires_after_1_hour(self, tmp_path):
        """LLMCache with ttl_hours=1 expires entries after 1 hour."""
        from src.llm_client.cache import LLMCache
        from src.llm_client.base import LLMRequest, LLMResponse

        cache = LLMCache(base_dir=str(tmp_path), provider="test", ttl_hours=1)

        # Create a request and response
        request = LLMRequest(prompt="test prompt", cache_key_prefix="test")
        response = LLMResponse(text="test response", provider="test", model="test-model")

        # Cache the response
        cache.set(request, response)

        # Verify it's cached
        assert cache.get(request) is not None

        # Manually modify the cached_at time to be 2 hours ago
        cache_files = list(cache.cache_dir.glob("*.json"))
        assert len(cache_files) == 1

        with open(cache_files[0], 'r') as f:
            data = json.load(f)

        data['cached_at'] = time.time() - (2 * 3600)  # 2 hours ago

        with open(cache_files[0], 'w') as f:
            json.dump(data, f)

        # Now the entry should be expired
        result = cache.get(request)
        assert result is None


# =============================================================================
# Test LLMClient Cache TTL Pass-Through
# =============================================================================

class TestLLMClientCacheTTL:
    """Tests for LLMClient passing TTL to cache."""

    def test_llm_client_accepts_cache_ttl_hours_parameter(self, tmp_path):
        """LLMClient.__init__ accepts cache_ttl_hours parameter."""
        from src.llm_client.base import LLMClient

        # Create a concrete implementation for testing
        class TestClient(LLMClient):
            @property
            def provider_name(self):
                return "test"

            def _call_api(self, request):
                return "test response"

        client = TestClient(api_key="test", model="test", cache_ttl_hours=48)
        assert client.cache_ttl_hours == 48

    def test_llm_client_cache_ttl_hours_default_is_24(self, tmp_path):
        """LLMClient cache_ttl_hours defaults to 24."""
        from src.llm_client.base import LLMClient

        class TestClient(LLMClient):
            @property
            def provider_name(self):
                return "test"

            def _call_api(self, request):
                return "test response"

        client = TestClient(api_key="test", model="test")
        assert client.cache_ttl_hours == 24

    def test_llm_client_passes_ttl_to_cache(self, tmp_path):
        """LLMClient.cache property passes ttl_hours to LLMCache."""
        from src.llm_client.base import LLMClient

        class TestClient(LLMClient):
            @property
            def provider_name(self):
                return "test"

            def _call_api(self, request):
                return "test response"

        client = TestClient(
            api_key="test",
            model="test",
            cache_dir=str(tmp_path),
            cache_ttl_hours=12
        )

        # Access cache property to trigger lazy initialization
        cache = client.cache

        # Verify the cache has the correct TTL
        assert cache.ttl_seconds == 12 * 3600


# =============================================================================
# Test Provider Clients TTL
# =============================================================================

class TestProviderClientsTTL:
    """Tests for provider clients accepting cache_ttl_hours."""

    def test_gemini_client_accepts_cache_ttl_hours(self, tmp_path):
        """GeminiClient.__init__ accepts cache_ttl_hours parameter."""
        from src.llm_client.providers.gemini import GeminiClient

        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel'):
                client = GeminiClient(
                    api_key="test-key",
                    model="test-model",
                    cache_dir=str(tmp_path),
                    cache_ttl_hours=6
                )
                assert client.cache_ttl_hours == 6

    def test_anthropic_client_accepts_cache_ttl_hours(self, tmp_path):
        """AnthropicClient.__init__ accepts cache_ttl_hours parameter."""
        from src.llm_client.providers.anthropic import AnthropicClient

        with patch('anthropic.Anthropic'):
            client = AnthropicClient(
                api_key="test-key",
                model="test-model",
                cache_dir=str(tmp_path),
                cache_ttl_hours=48
            )
            assert client.cache_ttl_hours == 48

    def test_ollama_client_accepts_cache_ttl_hours(self, tmp_path):
        """OllamaClient.__init__ accepts cache_ttl_hours parameter."""
        from src.llm_client.providers.ollama import OllamaClient

        client = OllamaClient(
            model="test-model",
            host="http://localhost:11434",
            cache_dir=str(tmp_path),
            cache_ttl_hours=72
        )
        assert client.cache_ttl_hours == 72


# =============================================================================
# Test Factory Functions
# =============================================================================

class TestFactoryCacheTTL:
    """Tests for factory functions using cache_ttl_hours."""

    def test_create_client_accepts_cache_ttl_hours(self, tmp_path):
        """create_client() accepts cache_ttl_hours parameter."""
        from src.llm_client.factory import create_client

        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel'):
                client = create_client(
                    provider="gemini",
                    api_key="test-key",
                    cache_dir=str(tmp_path),
                    cache_ttl_hours=36
                )
                assert client.cache_ttl_hours == 36

    def test_create_client_default_cache_ttl_is_24(self, tmp_path):
        """create_client() defaults cache_ttl_hours to 24."""
        from src.llm_client.factory import create_client

        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel'):
                client = create_client(
                    provider="gemini",
                    api_key="test-key",
                    cache_dir=str(tmp_path)
                )
                assert client.cache_ttl_hours == 24

    def test_create_client_from_config_uses_cache_ttl_hours(self, tmp_path):
        """create_client_from_config() extracts ttl_hours from config.llm.cache."""
        from src.llm_client.factory import create_client_from_config
        from src.config.sections.llm import LLMConfig, LLMCacheConfig

        # Create a mock config with custom TTL
        mock_llm_config = LLMConfig()
        mock_llm_config.cache = LLMCacheConfig(ttl_hours=1)
        mock_llm_config.provider = 'gemini'
        mock_llm_config.api_key = 'test-key'

        mock_config = Mock()
        mock_config.llm = mock_llm_config
        mock_config.cache_dir = str(tmp_path)

        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel'):
                client = create_client_from_config(mock_config)
                assert client.cache_ttl_hours == 1


# =============================================================================
# Test TTL=1 Hour Expiration
# =============================================================================

class TestTTL1HourExpiration:
    """Tests that cache_ttl_hours=1 expires entries after 1 hour."""

    def test_setting_cache_ttl_hours_1_expires_after_1_hour(self, tmp_path):
        """Setting cache_ttl_hours=1 expires entries after 1 hour."""
        from src.llm_client.cache import LLMCache
        from src.llm_client.base import LLMRequest, LLMResponse

        # Create cache with 1-hour TTL
        cache = LLMCache(base_dir=str(tmp_path), provider="test", ttl_hours=1)

        # Create and cache a response
        request = LLMRequest(prompt="test", cache_key_prefix="ttl_test")
        response = LLMResponse(text="result", provider="test", model="test")
        cache.set(request, response)

        # Entry should be retrievable immediately
        cached = cache.get(request)
        assert cached is not None
        assert cached['text'] == "result"

        # Manually expire the entry by modifying cached_at
        cache_files = list(cache.cache_dir.glob("*.json"))
        assert len(cache_files) == 1

        with open(cache_files[0], 'r') as f:
            data = json.load(f)

        # Set cached_at to 61 minutes ago (just over 1 hour)
        data['cached_at'] = time.time() - (61 * 60)

        with open(cache_files[0], 'w') as f:
            json.dump(data, f)

        # Entry should now be expired
        expired = cache.get(request)
        assert expired is None

    def test_cache_ttl_1_entry_valid_at_59_minutes(self, tmp_path):
        """Entry cached 59 minutes ago is still valid with ttl_hours=1."""
        from src.llm_client.cache import LLMCache
        from src.llm_client.base import LLMRequest, LLMResponse

        cache = LLMCache(base_dir=str(tmp_path), provider="test", ttl_hours=1)

        request = LLMRequest(prompt="test", cache_key_prefix="valid_test")
        response = LLMResponse(text="valid", provider="test", model="test")
        cache.set(request, response)

        # Modify to be 59 minutes old (just under 1 hour)
        cache_files = list(cache.cache_dir.glob("*.json"))
        with open(cache_files[0], 'r') as f:
            data = json.load(f)

        data['cached_at'] = time.time() - (59 * 60)

        with open(cache_files[0], 'w') as f:
            json.dump(data, f)

        # Entry should still be valid
        cached = cache.get(request)
        assert cached is not None


# =============================================================================
# Test Config Documentation
# =============================================================================

class TestCacheTTLDocumentation:
    """Tests that cache_ttl_hours is documented in config.yaml."""

    def test_config_yaml_has_llm_cache_ttl_hours(self):
        """config.yaml has llm.cache.ttl_hours setting."""
        config_path = Path(__file__).parent.parent / "config.yaml"
        content = config_path.read_text()

        # Check for ttl_hours in the llm.cache section
        assert "ttl_hours" in content
        assert "24" in content  # Default value
        assert "0 = never expire" in content  # Documentation


# =============================================================================
# Test Integration - Full Stack
# =============================================================================

class TestCacheTTLIntegration:
    """Integration tests for cache TTL through the full stack."""

    def test_end_to_end_ttl_configuration(self, tmp_path):
        """Test that TTL flows from config to cache correctly."""
        from src.llm_client.factory import create_client
        from src.llm_client.base import LLMRequest

        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel'):
                # Create client with custom TTL
                client = create_client(
                    provider="gemini",
                    api_key="test-key",
                    cache_dir=str(tmp_path),
                    cache_ttl_hours=6
                )

                # Verify TTL is stored on client
                assert client.cache_ttl_hours == 6

                # Access cache to trigger initialization
                cache = client.cache

                # Verify cache has correct TTL
                assert cache.ttl_seconds == 6 * 3600

    def test_ttl_0_never_expires(self, tmp_path):
        """Test that ttl_hours=0 results in entries that never expire."""
        from src.llm_client.cache import LLMCache
        from src.llm_client.base import LLMRequest, LLMResponse

        cache = LLMCache(base_dir=str(tmp_path), provider="test", ttl_hours=0)

        request = LLMRequest(prompt="test", cache_key_prefix="never_expire")
        response = LLMResponse(text="forever", provider="test", model="test")
        cache.set(request, response)

        # Make entry very old (1 year ago)
        cache_files = list(cache.cache_dir.glob("*.json"))
        with open(cache_files[0], 'r') as f:
            data = json.load(f)

        data['cached_at'] = time.time() - (365 * 24 * 3600)  # 1 year ago

        with open(cache_files[0], 'w') as f:
            json.dump(data, f)

        # Entry should still be valid
        cached = cache.get(request)
        assert cached is not None
        assert cached['text'] == "forever"
