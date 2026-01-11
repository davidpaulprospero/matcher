"""
Unit tests for LLM client base classes.
"""

import pytest
from src.llm_client.base import (
    ResponseFormat,
    LLMRequest,
    LLMResponse,
    LLMClient
)


class TestResponseFormat:
    """Test ResponseFormat enum."""

    def test_values(self):
        """Test enum values are correct."""
        assert ResponseFormat.TEXT.value == "text"
        assert ResponseFormat.JSON.value == "json"
        assert ResponseFormat.JSON_ARRAY.value == "json_array"

    def test_all_formats_present(self):
        """Test all expected formats are defined."""
        formats = [f.value for f in ResponseFormat]
        assert "text" in formats
        assert "json" in formats
        assert "json_array" in formats


class TestLLMRequest:
    """Test LLMRequest dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        request = LLMRequest(prompt="test prompt")

        assert request.prompt == "test prompt"
        assert request.system_prompt is None
        assert request.max_tokens == 2000
        assert request.temperature == 0.7
        assert request.response_format == ResponseFormat.TEXT
        assert request.timeout == 120
        assert request.batch_prompts is None
        assert request.images is None
        assert request.image_format == "jpeg"
        assert request.cache_key_prefix == "default"
        assert request.use_cache is True
        assert request.metadata == {}

    def test_custom_values(self):
        """Test custom values can be set."""
        request = LLMRequest(
            prompt="custom prompt",
            system_prompt="system",
            max_tokens=1000,
            temperature=0.5,
            response_format=ResponseFormat.JSON,
            timeout=60,
            cache_key_prefix="test",
            use_cache=False
        )

        assert request.prompt == "custom prompt"
        assert request.system_prompt == "system"
        assert request.max_tokens == 1000
        assert request.temperature == 0.5
        assert request.response_format == ResponseFormat.JSON
        assert request.timeout == 60
        assert request.cache_key_prefix == "test"
        assert request.use_cache is False

    def test_with_images(self):
        """Test request with images."""
        image_data = b"fake image data"
        request = LLMRequest(
            prompt="describe this",
            images=[image_data],
            image_format="png"
        )

        assert request.images == [image_data]
        assert request.image_format == "png"

    def test_with_metadata(self):
        """Test request with metadata."""
        metadata = {"source": "test", "version": 1}
        request = LLMRequest(
            prompt="test",
            metadata=metadata
        )

        assert request.metadata == metadata


class TestLLMResponse:
    """Test LLMResponse dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        response = LLMResponse(text="test response")

        assert response.text == "test response"
        assert response.parsed_data is None
        assert response.confidence == 1.0
        assert response.provider == ""
        assert response.model == ""
        assert response.cached is False
        assert response.request_time_ms == 0.0
        assert response.tokens_used is None
        assert response.metadata == {}

    def test_with_parsed_json(self):
        """Test response with parsed JSON data."""
        parsed = {"key": "value"}
        response = LLMResponse(
            text='{"key": "value"}',
            parsed_data=parsed
        )

        assert response.parsed_data == parsed

    def test_with_parsed_array(self):
        """Test response with parsed JSON array."""
        parsed = [{"id": 1}, {"id": 2}]
        response = LLMResponse(
            text='[{"id": 1}, {"id": 2}]',
            parsed_data=parsed
        )

        assert response.parsed_data == parsed

    def test_cached_response(self):
        """Test cached response properties."""
        response = LLMResponse(
            text="cached text",
            provider="gemini",
            model="gemini-2.0-flash",
            cached=True
        )

        assert response.cached is True
        assert response.provider == "gemini"
        assert response.model == "gemini-2.0-flash"


class MockLLMClient(LLMClient):
    """Mock LLM client for testing base class functionality."""

    @property
    def provider_name(self) -> str:
        return "mock"

    def _call_api(self, request: LLMRequest) -> str:
        """Return mock response based on prompt."""
        if "json" in request.prompt.lower():
            return '{"result": "success"}'
        elif "array" in request.prompt.lower():
            return '[{"id": 1}, {"id": 2}]'
        else:
            return "Mock response"


class TestLLMClient:
    """Test LLMClient base class."""

    def test_initialization(self):
        """Test client can be initialized."""
        client = MockLLMClient(api_key="test_key", model="test_model")

        assert client.api_key == "test_key"
        assert client.model == "test_model"
        assert client.cache_dir == ".cache/llm_responses"
        assert client.provider_name == "mock"

    def test_generate_text(self):
        """Test generating text response."""
        client = MockLLMClient(api_key="test", model="test")
        request = LLMRequest(
            prompt="Hello",
            response_format=ResponseFormat.TEXT,
            use_cache=False  # Disable cache for testing
        )

        response = client.generate(request)

        assert response.text == "Mock response"
        assert response.parsed_data is None
        assert response.provider == "mock"
        assert response.model == "test"
        assert response.cached is False

    def test_generate_json(self):
        """Test generating and parsing JSON response."""
        client = MockLLMClient(api_key="test", model="test")
        request = LLMRequest(
            prompt="Return JSON",
            response_format=ResponseFormat.JSON,
            use_cache=False
        )

        response = client.generate(request)

        assert response.text == '{"result": "success"}'
        assert response.parsed_data == {"result": "success"}

    def test_generate_json_array(self):
        """Test generating and parsing JSON array response."""
        client = MockLLMClient(api_key="test", model="test")
        request = LLMRequest(
            prompt="Return array",
            response_format=ResponseFormat.JSON_ARRAY,
            use_cache=False
        )

        response = client.generate(request)

        assert response.parsed_data == [{"id": 1}, {"id": 2}]

    def test_request_time_tracked(self):
        """Test that request time is tracked."""
        client = MockLLMClient(api_key="test", model="test")
        request = LLMRequest(prompt="test", use_cache=False)

        response = client.generate(request)

        assert response.request_time_ms >= 0

    def test_cache_property(self):
        """Test cache property is lazy-loaded."""
        client = MockLLMClient(api_key="test", model="test")

        # Cache should be None initially
        assert client._cache is None

        # Accessing cache should initialize it
        cache = client.cache
        assert cache is not None
        assert cache.provider == "mock"
