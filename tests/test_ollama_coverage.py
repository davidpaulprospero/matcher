"""
Comprehensive tests for src/llm_client/providers/ollama.py

Tests OllamaClient class:
- Initialization
- Provider name
- API calls
- Error handling (timeout, connection, HTTP errors)
- Temperature settings
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# Mock ResponseFormat
class MockResponseFormat(Enum):
    TEXT = "text"
    JSON = "json"


@dataclass
class MockLLMRequest:
    """Mock LLM request for testing"""
    prompt: str
    system_prompt: str = ""
    max_tokens: int = 1000
    temperature: float = 0.7
    response_format: MockResponseFormat = MockResponseFormat.TEXT
    cache_key_prefix: str = "test"
    images: list = None
    timeout: int = 60


# ============================================================================
# Test Initialization
# ============================================================================

class TestOllamaClientInit:
    """Test OllamaClient initialization"""

    def test_init_default_values(self, tmp_path):
        """Test initialization with default values"""
        from src.llm_client.providers.ollama import OllamaClient

        client = OllamaClient(cache_dir=str(tmp_path / "cache"))

        assert client.model == "llama3.2"
        assert client.host == "http://localhost:11434"
        assert client.provider_name == "ollama"

    def test_init_custom_model(self, tmp_path):
        """Test initialization with custom model"""
        from src.llm_client.providers.ollama import OllamaClient

        client = OllamaClient(
            model="mistral",
            cache_dir=str(tmp_path / "cache")
        )

        assert client.model == "mistral"

    def test_init_custom_host(self, tmp_path):
        """Test initialization with custom host"""
        from src.llm_client.providers.ollama import OllamaClient

        client = OllamaClient(
            host="http://192.168.1.100:11434",
            cache_dir=str(tmp_path / "cache")
        )

        assert client.host == "http://192.168.1.100:11434"

    def test_init_strips_trailing_slash(self, tmp_path):
        """Test that trailing slash is stripped from host"""
        from src.llm_client.providers.ollama import OllamaClient

        client = OllamaClient(
            host="http://localhost:11434/",
            cache_dir=str(tmp_path / "cache")
        )

        assert client.host == "http://localhost:11434"


# ============================================================================
# Test API Calls
# ============================================================================

class TestOllamaAPICalls:
    """Test Ollama API calls"""

    def test_call_api_success(self, tmp_path):
        """Test successful API call"""
        from src.llm_client.providers.ollama import OllamaClient

        client = OllamaClient(cache_dir=str(tmp_path / "cache"))

        mock_response = Mock()
        mock_response.json.return_value = {"response": "Hello from Ollama!"}
        mock_response.raise_for_status = Mock()

        with patch('requests.post', return_value=mock_response) as mock_post:
            request = MockLLMRequest(prompt="Hello")
            result = client._call_api(request)

            assert result == "Hello from Ollama!"
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "llama3.2" in str(call_args)
            assert "Hello" in str(call_args)

    def test_call_api_with_custom_temperature(self, tmp_path):
        """Test API call includes temperature when not default"""
        from src.llm_client.providers.ollama import OllamaClient

        client = OllamaClient(cache_dir=str(tmp_path / "cache"))

        mock_response = Mock()
        mock_response.json.return_value = {"response": "Response"}
        mock_response.raise_for_status = Mock()

        with patch('requests.post', return_value=mock_response) as mock_post:
            request = MockLLMRequest(prompt="Test", temperature=0.5)
            client._call_api(request)

            # Check that temperature was included in payload
            call_kwargs = mock_post.call_args[1]
            assert call_kwargs['json']['temperature'] == 0.5

    def test_call_api_default_temperature_not_included(self, tmp_path):
        """Test API call doesn't include temperature when default 0.7"""
        from src.llm_client.providers.ollama import OllamaClient

        client = OllamaClient(cache_dir=str(tmp_path / "cache"))

        mock_response = Mock()
        mock_response.json.return_value = {"response": "Response"}
        mock_response.raise_for_status = Mock()

        with patch('requests.post', return_value=mock_response) as mock_post:
            request = MockLLMRequest(prompt="Test", temperature=0.7)
            client._call_api(request)

            # Check that temperature was NOT included (default)
            call_kwargs = mock_post.call_args[1]
            assert 'temperature' not in call_kwargs['json']

    def test_call_api_empty_response(self, tmp_path):
        """Test handling of empty response"""
        from src.llm_client.providers.ollama import OllamaClient

        client = OllamaClient(cache_dir=str(tmp_path / "cache"))

        mock_response = Mock()
        mock_response.json.return_value = {}  # Missing 'response' key
        mock_response.raise_for_status = Mock()

        with patch('requests.post', return_value=mock_response):
            request = MockLLMRequest(prompt="Test")
            result = client._call_api(request)

            assert result == ""  # Should return empty string


# ============================================================================
# Test Error Handling
# ============================================================================

class TestOllamaErrorHandling:
    """Test Ollama error handling"""

    def test_requests_import_error(self, tmp_path):
        """Test handling when requests package not installed"""
        from src.llm_client.providers.ollama import OllamaClient
        from src.llm_client.exceptions import LLMProviderError

        client = OllamaClient(cache_dir=str(tmp_path / "cache"))

        # Mock import failure for requests
        with patch.dict('sys.modules', {'requests': None}):
            with patch('builtins.__import__', side_effect=ImportError("No module named 'requests'")):
                with pytest.raises(LLMProviderError) as exc_info:
                    request = MockLLMRequest(prompt="Test")
                    client._call_api(request)

                assert "requests package not installed" in str(exc_info.value)

    def test_timeout_error(self, tmp_path):
        """Test handling of timeout errors"""
        from src.llm_client.providers.ollama import OllamaClient
        from src.llm_client.exceptions import LLMTimeoutError
        import requests

        client = OllamaClient(cache_dir=str(tmp_path / "cache"))

        with patch('requests.post', side_effect=requests.exceptions.Timeout("Connection timed out")):
            with pytest.raises(LLMTimeoutError) as exc_info:
                request = MockLLMRequest(prompt="Test")
                client._call_api(request)

            assert "timed out" in str(exc_info.value).lower()

    def test_connection_error(self, tmp_path):
        """Test handling of connection errors"""
        from src.llm_client.providers.ollama import OllamaClient
        from src.llm_client.exceptions import LLMProviderError
        import requests

        client = OllamaClient(cache_dir=str(tmp_path / "cache"))

        with patch('requests.post', side_effect=requests.exceptions.ConnectionError("Connection refused")):
            with pytest.raises(LLMProviderError) as exc_info:
                request = MockLLMRequest(prompt="Test")
                client._call_api(request)

            assert "Failed to connect" in str(exc_info.value)
            assert "Is Ollama running?" in str(exc_info.value)

    def test_http_404_model_not_found(self, tmp_path):
        """Test handling of 404 error (model not found)"""
        from src.llm_client.providers.ollama import OllamaClient
        from src.llm_client.exceptions import LLMProviderError
        import requests

        client = OllamaClient(model="nonexistent-model", cache_dir=str(tmp_path / "cache"))

        mock_response = Mock()
        mock_response.status_code = 404
        http_error = requests.exceptions.HTTPError(response=mock_response)
        http_error.response = mock_response

        with patch('requests.post', side_effect=http_error):
            with pytest.raises(LLMProviderError) as exc_info:
                request = MockLLMRequest(prompt="Test")
                client._call_api(request)

            assert "not found" in str(exc_info.value).lower()
            assert "ollama pull" in str(exc_info.value)
            assert "nonexistent-model" in str(exc_info.value)

    def test_http_other_error(self, tmp_path):
        """Test handling of other HTTP errors"""
        from src.llm_client.providers.ollama import OllamaClient
        from src.llm_client.exceptions import LLMProviderError
        import requests

        client = OllamaClient(cache_dir=str(tmp_path / "cache"))

        mock_response = Mock()
        mock_response.status_code = 500
        http_error = requests.exceptions.HTTPError(response=mock_response)
        http_error.response = mock_response

        with patch('requests.post', side_effect=http_error):
            with pytest.raises(LLMProviderError) as exc_info:
                request = MockLLMRequest(prompt="Test")
                client._call_api(request)

            assert "API error" in str(exc_info.value)

    def test_http_error_without_response(self, tmp_path):
        """Test handling of HTTP error with response=None triggers AttributeError path"""
        from src.llm_client.providers.ollama import OllamaClient
        from src.llm_client.exceptions import LLMProviderError
        import requests

        client = OllamaClient(cache_dir=str(tmp_path / "cache"))

        # Create HTTP error with response=None (hasattr returns True but value is None)
        http_error = requests.exceptions.HTTPError("HTTP Error")
        http_error.response = None  # This causes the AttributeError path

        with patch('requests.post', side_effect=http_error):
            # This will hit the generic exception handler due to AttributeError
            with pytest.raises((LLMProviderError, AttributeError)):
                request = MockLLMRequest(prompt="Test")
                client._call_api(request)

    def test_generic_exception(self, tmp_path):
        """Test handling of generic exceptions"""
        from src.llm_client.providers.ollama import OllamaClient
        from src.llm_client.exceptions import LLMProviderError

        client = OllamaClient(cache_dir=str(tmp_path / "cache"))

        with patch('requests.post', side_effect=ValueError("Unexpected error")):
            with pytest.raises(LLMProviderError) as exc_info:
                request = MockLLMRequest(prompt="Test")
                client._call_api(request)

            assert "Ollama error" in str(exc_info.value)


# ============================================================================
# Test Integration with Base Client
# ============================================================================

class TestOllamaIntegration:
    """Test Ollama integration with base client"""

    def test_provider_name_property(self, tmp_path):
        """Test provider_name property returns 'ollama'"""
        from src.llm_client.providers.ollama import OllamaClient

        client = OllamaClient(cache_dir=str(tmp_path / "cache"))

        assert client.provider_name == "ollama"

    def test_timeout_passed_to_request(self, tmp_path):
        """Test that timeout is passed to requests.post"""
        from src.llm_client.providers.ollama import OllamaClient

        client = OllamaClient(cache_dir=str(tmp_path / "cache"))

        mock_response = Mock()
        mock_response.json.return_value = {"response": "OK"}
        mock_response.raise_for_status = Mock()

        with patch('requests.post', return_value=mock_response) as mock_post:
            request = MockLLMRequest(prompt="Test", timeout=120)
            client._call_api(request)

            call_kwargs = mock_post.call_args[1]
            assert call_kwargs['timeout'] == 120

    def test_correct_endpoint_used(self, tmp_path):
        """Test that correct API endpoint is used"""
        from src.llm_client.providers.ollama import OllamaClient

        client = OllamaClient(
            host="http://custom-host:11434",
            cache_dir=str(tmp_path / "cache")
        )

        mock_response = Mock()
        mock_response.json.return_value = {"response": "OK"}
        mock_response.raise_for_status = Mock()

        with patch('requests.post', return_value=mock_response) as mock_post:
            request = MockLLMRequest(prompt="Test")
            client._call_api(request)

            call_url = mock_post.call_args[0][0]
            assert call_url == "http://custom-host:11434/api/generate"


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestOllamaEdgeCases:
    """Test edge cases"""

    def test_empty_prompt(self, tmp_path):
        """Test API call with empty prompt"""
        from src.llm_client.providers.ollama import OllamaClient

        client = OllamaClient(cache_dir=str(tmp_path / "cache"))

        mock_response = Mock()
        mock_response.json.return_value = {"response": ""}
        mock_response.raise_for_status = Mock()

        with patch('requests.post', return_value=mock_response) as mock_post:
            request = MockLLMRequest(prompt="")
            result = client._call_api(request)

            assert result == ""
            call_kwargs = mock_post.call_args[1]
            assert call_kwargs['json']['prompt'] == ""

    def test_long_prompt(self, tmp_path):
        """Test API call with very long prompt"""
        from src.llm_client.providers.ollama import OllamaClient

        client = OllamaClient(cache_dir=str(tmp_path / "cache"))

        long_prompt = "A" * 10000

        mock_response = Mock()
        mock_response.json.return_value = {"response": "OK"}
        mock_response.raise_for_status = Mock()

        with patch('requests.post', return_value=mock_response) as mock_post:
            request = MockLLMRequest(prompt=long_prompt)
            result = client._call_api(request)

            assert result == "OK"
            call_kwargs = mock_post.call_args[1]
            assert len(call_kwargs['json']['prompt']) == 10000

    def test_unicode_prompt(self, tmp_path):
        """Test API call with unicode characters"""
        from src.llm_client.providers.ollama import OllamaClient

        client = OllamaClient(cache_dir=str(tmp_path / "cache"))

        mock_response = Mock()
        mock_response.json.return_value = {"response": "OK"}
        mock_response.raise_for_status = Mock()

        with patch('requests.post', return_value=mock_response) as mock_post:
            request = MockLLMRequest(prompt="Test with unicode: ")
            client._call_api(request)

            call_kwargs = mock_post.call_args[1]
            assert "" in call_kwargs['json']['prompt']

    def test_stream_always_false(self, tmp_path):
        """Test that stream is always set to False"""
        from src.llm_client.providers.ollama import OllamaClient

        client = OllamaClient(cache_dir=str(tmp_path / "cache"))

        mock_response = Mock()
        mock_response.json.return_value = {"response": "OK"}
        mock_response.raise_for_status = Mock()

        with patch('requests.post', return_value=mock_response) as mock_post:
            request = MockLLMRequest(prompt="Test")
            client._call_api(request)

            call_kwargs = mock_post.call_args[1]
            assert call_kwargs['json']['stream'] is False
