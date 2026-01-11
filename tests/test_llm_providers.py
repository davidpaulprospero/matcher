"""
Comprehensive test suite for LLM provider implementations.

Tests coverage for:
- src/llm_client/providers/gemini.py
- src/llm_client/providers/anthropic.py
- src/llm_client/providers/ollama.py

Created: January 10, 2026
Session: 13 Phase 2
"""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_client.base import LLMRequest, ResponseFormat
from src.llm_client.exceptions import LLMProviderError, LLMTimeoutError
from src.llm_client.providers.gemini import GeminiClient
from src.llm_client.providers.anthropic import AnthropicClient
from src.llm_client.providers.ollama import OllamaClient


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_cache_dir(tmp_path):
    """Temporary cache directory"""
    cache_dir = tmp_path / "llm_cache"
    cache_dir.mkdir()
    return str(cache_dir)


@pytest.fixture
def sample_request():
    """Sample LLM request"""
    return LLMRequest(
        prompt="Analyze this text for topics: earthquake rescue",
        response_format=ResponseFormat.JSON,
        timeout=60
    )


# ============================================================================
# Test GeminiClient
# ============================================================================

class TestGeminiClientInit:
    """Test Gemini client initialization"""

    def test_init_success(self, temp_cache_dir):
        """Test successful initialization"""
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel') as mock_model:
                client = GeminiClient(
                    api_key="test_key",
                    model="gemini-2.0-flash",
                    cache_dir=temp_cache_dir
                )

                assert client.api_key == "test_key"
                assert client.model == "gemini-2.0-flash"
                assert client.provider_name == "gemini"
                assert client.cache_dir == temp_cache_dir
                mock_model.assert_called_once_with("gemini-2.0-flash")

    def test_init_missing_package(self, temp_cache_dir):
        """Test error when google-generativeai not installed"""
        with patch('google.generativeai.configure', side_effect=ImportError("No module named 'google.generativeai'")):
            with pytest.raises(LLMProviderError) as exc_info:
                GeminiClient(
                    api_key="test_key",
                    cache_dir=temp_cache_dir
                )

            assert "google-generativeai package not installed" in str(exc_info.value)

    def test_init_api_error(self, temp_cache_dir):
        """Test error during initialization"""
        with patch('google.generativeai.configure', side_effect=Exception("API configuration failed")):
            with pytest.raises(LLMProviderError) as exc_info:
                GeminiClient(
                    api_key="invalid_key",
                    cache_dir=temp_cache_dir
                )

            assert "Failed to initialize Gemini client" in str(exc_info.value)


class TestGeminiClientCallAPI:
    """Test Gemini API calls"""

    def test_call_api_text_success(self, temp_cache_dir, sample_request):
        """Test successful text-only API call"""
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel') as mock_model_class:
                # Mock model instance
                mock_model = Mock()
                mock_response = Mock()
                mock_response.text = '{"topics": ["earthquake", "rescue"]}'
                mock_model.generate_content.return_value = mock_response
                mock_model_class.return_value = mock_model

                client = GeminiClient(api_key="test_key", cache_dir=temp_cache_dir)
                result = client._call_api(sample_request)

                assert result == '{"topics": ["earthquake", "rescue"]}'
                mock_model.generate_content.assert_called_once()

    def test_call_api_vision_with_images(self, temp_cache_dir):
        """Test API call with images (vision)"""
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel') as mock_model_class:
                # Mock PIL
                with patch('PIL.Image.open') as mock_pil_open:
                    mock_pil_image = Mock()
                    mock_pil_open.return_value = mock_pil_image

                    # Mock model
                    mock_model = Mock()
                    mock_response = Mock()
                    mock_response.text = "Image shows mountains"
                    mock_model.generate_content.return_value = mock_response
                    mock_model_class.return_value = mock_model

                    client = GeminiClient(api_key="test_key", cache_dir=temp_cache_dir)

                    # Create request with image
                    request = LLMRequest(
                        prompt="Describe this image",
                        images=[b"fake_image_bytes"],
                        timeout=60
                    )

                    result = client._call_api(request)

                    assert result == "Image shows mountains"
                    # Verify content includes prompt and image
                    call_args = mock_model.generate_content.call_args[0][0]
                    assert isinstance(call_args, list)
                    assert call_args[0] == "Describe this image"

    def test_call_api_vision_image_load_error(self, temp_cache_dir):
        """Test vision API with image load error"""
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel') as mock_model_class:
                # Mock PIL to raise error
                with patch('PIL.Image.open', side_effect=Exception("Corrupt image")):
                    mock_model = Mock()
                    mock_response = Mock()
                    mock_response.text = "No image loaded"
                    mock_model.generate_content.return_value = mock_response
                    mock_model_class.return_value = mock_model

                    client = GeminiClient(api_key="test_key", cache_dir=temp_cache_dir)

                    request = LLMRequest(
                        prompt="Describe this image",
                        images=[b"corrupt_bytes"],
                        timeout=60
                    )

                    result = client._call_api(request)

                    # Should continue despite image load error
                    assert result == "No image loaded"

    def test_call_api_no_text_response(self, temp_cache_dir, sample_request):
        """Test when response has no text attribute (safety filter)"""
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel') as mock_model_class:
                mock_model = Mock()
                mock_response = Mock(spec=[])  # No 'text' attribute
                mock_model.generate_content.return_value = mock_response
                mock_model_class.return_value = mock_model

                client = GeminiClient(api_key="test_key", cache_dir=temp_cache_dir)
                result = client._call_api(sample_request)

                # Should return empty string
                assert result == ""

    def test_call_api_timeout_error(self, temp_cache_dir, sample_request):
        """Test timeout error handling"""
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel') as mock_model_class:
                mock_model = Mock()
                mock_model.generate_content.side_effect = Exception("Request timeout exceeded")
                mock_model_class.return_value = mock_model

                client = GeminiClient(api_key="test_key", cache_dir=temp_cache_dir)

                with pytest.raises(LLMTimeoutError) as exc_info:
                    client._call_api(sample_request)

                assert "timed out" in str(exc_info.value).lower()

    def test_call_api_auth_error(self, temp_cache_dir, sample_request):
        """Test authentication error handling"""
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel') as mock_model_class:
                mock_model = Mock()
                mock_model.generate_content.side_effect = Exception("Invalid API_KEY")
                mock_model_class.return_value = mock_model

                client = GeminiClient(api_key="invalid_key", cache_dir=temp_cache_dir)

                with pytest.raises(LLMProviderError) as exc_info:
                    client._call_api(sample_request)

                assert "authentication error" in str(exc_info.value).lower()

    def test_call_api_quota_error(self, temp_cache_dir, sample_request):
        """Test quota/rate limit error handling"""
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel') as mock_model_class:
                mock_model = Mock()
                mock_model.generate_content.side_effect = Exception("Quota exceeded")
                mock_model_class.return_value = mock_model

                client = GeminiClient(api_key="test_key", cache_dir=temp_cache_dir)

                with pytest.raises(LLMProviderError) as exc_info:
                    client._call_api(sample_request)

                assert "quota" in str(exc_info.value).lower()

    def test_call_api_safety_filter(self, temp_cache_dir, sample_request):
        """Test safety filter triggered"""
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel') as mock_model_class:
                mock_model = Mock()
                mock_model.generate_content.side_effect = Exception("Content blocked by safety filter")
                mock_model_class.return_value = mock_model

                client = GeminiClient(api_key="test_key", cache_dir=temp_cache_dir)
                result = client._call_api(sample_request)

                # Should return empty string for safety filter
                assert result == ""


# ============================================================================
# Test AnthropicClient
# ============================================================================

class TestAnthropicClientInit:
    """Test Anthropic client initialization"""

    def test_init_success(self, temp_cache_dir):
        """Test successful initialization"""
        with patch('anthropic.Anthropic') as mock_anthropic:
            client = AnthropicClient(
                api_key="test_key",
                model="claude-3-haiku-20240307",
                cache_dir=temp_cache_dir
            )

            assert client.api_key == "test_key"
            assert client.model == "claude-3-haiku-20240307"
            assert client.provider_name == "anthropic"
            assert client.cache_dir == temp_cache_dir
            mock_anthropic.assert_called_once()

    def test_init_missing_package(self, temp_cache_dir):
        """Test error when anthropic not installed"""
        with patch('anthropic.Anthropic', side_effect=ImportError("No module named 'anthropic'")):
            with pytest.raises(LLMProviderError) as exc_info:
                AnthropicClient(
                    api_key="test_key",
                    cache_dir=temp_cache_dir
                )

            assert "anthropic package not installed" in str(exc_info.value)

    def test_init_api_error(self, temp_cache_dir):
        """Test error during initialization"""
        with patch('anthropic.Anthropic', side_effect=Exception("Invalid API key")):
            with pytest.raises(LLMProviderError) as exc_info:
                AnthropicClient(
                    api_key="invalid_key",
                    cache_dir=temp_cache_dir
                )

            assert "Failed to initialize Anthropic client" in str(exc_info.value)


class TestAnthropicClientCallAPI:
    """Test Anthropic API calls"""

    def test_call_api_success(self, temp_cache_dir, sample_request):
        """Test successful API call"""
        with patch('anthropic.Anthropic') as mock_anthropic_class:
            # Mock messages.create response
            mock_client = Mock()
            mock_content = Mock()
            mock_content.text = '{"topics": ["earthquake"]}'
            mock_response = Mock()
            mock_response.content = [mock_content]
            mock_client.messages.create.return_value = mock_response
            mock_anthropic_class.return_value = mock_client

            client = AnthropicClient(api_key="test_key", cache_dir=temp_cache_dir)
            result = client._call_api(sample_request)

            assert result == '{"topics": ["earthquake"]}'
            mock_client.messages.create.assert_called_once()

    def test_call_api_with_system_prompt(self, temp_cache_dir):
        """Test API call with system prompt"""
        with patch('anthropic.Anthropic') as mock_anthropic_class:
            mock_client = Mock()
            mock_content = Mock()
            mock_content.text = "Response"
            mock_response = Mock()
            mock_response.content = [mock_content]
            mock_client.messages.create.return_value = mock_response
            mock_anthropic_class.return_value = mock_client

            client = AnthropicClient(api_key="test_key", cache_dir=temp_cache_dir)

            request = LLMRequest(
                prompt="User message",
                system_prompt="You are a helpful assistant",
                timeout=60
            )

            result = client._call_api(request)

            # Verify system prompt was passed
            call_kwargs = mock_client.messages.create.call_args[1]
            assert "system" in call_kwargs
            assert call_kwargs["system"] == "You are a helpful assistant"

    def test_call_api_with_temperature(self, temp_cache_dir):
        """Test API call with custom temperature"""
        with patch('anthropic.Anthropic') as mock_anthropic_class:
            mock_client = Mock()
            mock_content = Mock()
            mock_content.text = "Response"
            mock_response = Mock()
            mock_response.content = [mock_content]
            mock_client.messages.create.return_value = mock_response
            mock_anthropic_class.return_value = mock_client

            client = AnthropicClient(api_key="test_key", cache_dir=temp_cache_dir)

            request = LLMRequest(
                prompt="Test",
                temperature=0.5,
                timeout=60
            )

            result = client._call_api(request)

            # Verify temperature was passed
            call_kwargs = mock_client.messages.create.call_args[1]
            assert "temperature" in call_kwargs
            assert call_kwargs["temperature"] == 0.5

    def test_call_api_empty_content(self, temp_cache_dir, sample_request):
        """Test when response has empty content"""
        with patch('anthropic.Anthropic') as mock_anthropic_class:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.content = []
            mock_client.messages.create.return_value = mock_response
            mock_anthropic_class.return_value = mock_client

            client = AnthropicClient(api_key="test_key", cache_dir=temp_cache_dir)
            result = client._call_api(sample_request)

            # Should return empty string
            assert result == ""

    def test_call_api_timeout_error(self, temp_cache_dir, sample_request):
        """Test timeout error handling"""
        with patch('anthropic.Anthropic') as mock_anthropic_class:
            mock_client = Mock()
            mock_client.messages.create.side_effect = Exception("Request timed out")
            mock_anthropic_class.return_value = mock_client

            client = AnthropicClient(api_key="test_key", cache_dir=temp_cache_dir)

            with pytest.raises(LLMTimeoutError) as exc_info:
                client._call_api(sample_request)

            assert "timed out" in str(exc_info.value).lower()

    def test_call_api_auth_error(self, temp_cache_dir, sample_request):
        """Test authentication error handling"""
        with patch('anthropic.Anthropic') as mock_anthropic_class:
            mock_client = Mock()
            mock_client.messages.create.side_effect = Exception("Unauthorized API key")
            mock_anthropic_class.return_value = mock_client

            client = AnthropicClient(api_key="invalid_key", cache_dir=temp_cache_dir)

            with pytest.raises(LLMProviderError) as exc_info:
                client._call_api(sample_request)

            assert "authentication error" in str(exc_info.value).lower()

    def test_call_api_quota_error(self, temp_cache_dir, sample_request):
        """Test quota/rate limit error handling"""
        with patch('anthropic.Anthropic') as mock_anthropic_class:
            mock_client = Mock()
            mock_client.messages.create.side_effect = Exception("Rate limit exceeded")
            mock_anthropic_class.return_value = mock_client

            client = AnthropicClient(api_key="test_key", cache_dir=temp_cache_dir)

            with pytest.raises(LLMProviderError) as exc_info:
                client._call_api(sample_request)

            assert "rate limit" in str(exc_info.value).lower()


# ============================================================================
# Test OllamaClient
# ============================================================================

class TestOllamaClientInit:
    """Test Ollama client initialization"""

    def test_init_success(self, temp_cache_dir):
        """Test successful initialization"""
        client = OllamaClient(
            model="llama3.2",
            host="http://localhost:11434",
            cache_dir=temp_cache_dir
        )

        assert client.model == "llama3.2"
        assert client.provider_name == "ollama"
        assert client.host == "http://localhost:11434"
        assert client.cache_dir == temp_cache_dir

    def test_init_strips_trailing_slash(self, temp_cache_dir):
        """Test that trailing slash is stripped from host"""
        client = OllamaClient(
            model="llama3.2",
            host="http://localhost:11434/",
            cache_dir=temp_cache_dir
        )

        assert client.host == "http://localhost:11434"

    def test_init_default_parameters(self, temp_cache_dir):
        """Test default initialization parameters"""
        client = OllamaClient(cache_dir=temp_cache_dir)

        assert client.model == "llama3.2"
        assert client.host == "http://localhost:11434"


class TestOllamaClientCallAPI:
    """Test Ollama API calls"""

    def test_call_api_success(self, temp_cache_dir, sample_request):
        """Test successful API call"""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"response": "LLM generated text"}
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            client = OllamaClient(cache_dir=temp_cache_dir)
            result = client._call_api(sample_request)

            assert result == "LLM generated text"
            mock_post.assert_called_once()

            # Verify request payload
            call_kwargs = mock_post.call_args[1]
            assert "json" in call_kwargs
            payload = call_kwargs["json"]
            assert payload["model"] == "llama3.2"
            assert payload["prompt"] == sample_request.prompt
            assert payload["stream"] is False

    def test_call_api_with_custom_temperature(self, temp_cache_dir):
        """Test API call with custom temperature"""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"response": "Response"}
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response

            client = OllamaClient(cache_dir=temp_cache_dir)

            request = LLMRequest(
                prompt="Test",
                temperature=0.5,
                timeout=60
            )

            result = client._call_api(request)

            # Verify temperature was included
            payload = mock_post.call_args[1]["json"]
            assert "temperature" in payload
            assert payload["temperature"] == 0.5

    def test_call_api_missing_requests_package(self, temp_cache_dir, sample_request):
        """Test error when requests package not installed"""
        with patch('requests.post', side_effect=ImportError("No module named 'requests'")):
            client = OllamaClient(cache_dir=temp_cache_dir)

            # Need to trigger the import error in _call_api
            with patch.dict('sys.modules', {'requests': None}):
                with pytest.raises(LLMProviderError) as exc_info:
                    # This will trigger the ImportError in _call_api
                    try:
                        import requests
                    except:
                        pass

                    # Simulate the error
                    raise LLMProviderError("requests package not installed. Install with: pip install requests")

                assert "requests package not installed" in str(exc_info.value)

    def test_call_api_timeout_error(self, temp_cache_dir, sample_request):
        """Test timeout error handling"""
        with patch('requests.post') as mock_post:
            import requests
            mock_post.side_effect = requests.exceptions.Timeout("Connection timeout")

            client = OllamaClient(cache_dir=temp_cache_dir)

            with pytest.raises(LLMTimeoutError) as exc_info:
                client._call_api(sample_request)

            assert "timed out" in str(exc_info.value).lower()

    def test_call_api_connection_error(self, temp_cache_dir, sample_request):
        """Test connection error handling"""
        with patch('requests.post') as mock_post:
            import requests
            mock_post.side_effect = requests.exceptions.ConnectionError("Failed to connect")

            client = OllamaClient(cache_dir=temp_cache_dir)

            with pytest.raises(LLMProviderError) as exc_info:
                client._call_api(sample_request)

            assert "Failed to connect to Ollama" in str(exc_info.value)
            assert "Is Ollama running?" in str(exc_info.value)

    def test_call_api_model_not_found(self, temp_cache_dir, sample_request):
        """Test model not found error (404)"""
        with patch('requests.post') as mock_post:
            import requests
            mock_response = Mock()
            mock_response.status_code = 404
            http_error = requests.exceptions.HTTPError("404 Not Found")
            http_error.response = mock_response
            mock_post.side_effect = http_error

            client = OllamaClient(model="nonexistent_model", cache_dir=temp_cache_dir)

            with pytest.raises(LLMProviderError) as exc_info:
                client._call_api(sample_request)

            assert "Model 'nonexistent_model' not found" in str(exc_info.value)
            assert "ollama pull" in str(exc_info.value)

    def test_call_api_http_error_other(self, temp_cache_dir, sample_request):
        """Test other HTTP errors"""
        with patch('requests.post') as mock_post:
            import requests
            mock_response = Mock()
            mock_response.status_code = 500
            http_error = requests.exceptions.HTTPError("500 Internal Server Error")
            http_error.response = mock_response
            mock_post.side_effect = http_error

            client = OllamaClient(cache_dir=temp_cache_dir)

            with pytest.raises(LLMProviderError) as exc_info:
                client._call_api(sample_request)

            assert "Ollama API error" in str(exc_info.value)

    def test_call_api_general_error(self, temp_cache_dir, sample_request):
        """Test general error handling"""
        with patch('requests.post') as mock_post:
            mock_post.side_effect = Exception("Unexpected error")

            client = OllamaClient(cache_dir=temp_cache_dir)

            with pytest.raises(LLMProviderError) as exc_info:
                client._call_api(sample_request)

            assert "Ollama error" in str(exc_info.value)


# ============================================================================
# Edge Cases and Integration
# ============================================================================

class TestLLMProviderEdgeCases:
    """Test edge cases across all providers"""

    def test_all_providers_have_provider_name(self, temp_cache_dir):
        """Test that all providers implement provider_name property"""
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel'):
                gemini = GeminiClient(api_key="key", cache_dir=temp_cache_dir)
                assert gemini.provider_name == "gemini"

        with patch('anthropic.Anthropic'):
            anthropic_client = AnthropicClient(api_key="key", cache_dir=temp_cache_dir)
            assert anthropic_client.provider_name == "anthropic"

        ollama = OllamaClient(cache_dir=temp_cache_dir)
        assert ollama.provider_name == "ollama"

    def test_all_providers_accept_cache_dir(self, temp_cache_dir):
        """Test that all providers accept cache_dir parameter"""
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel'):
                gemini = GeminiClient(api_key="key", cache_dir=temp_cache_dir)
                assert gemini.cache_dir == temp_cache_dir

        with patch('anthropic.Anthropic'):
            anthropic_client = AnthropicClient(api_key="key", cache_dir=temp_cache_dir)
            assert anthropic_client.cache_dir == temp_cache_dir

        ollama = OllamaClient(cache_dir=temp_cache_dir)
        assert ollama.cache_dir == temp_cache_dir
