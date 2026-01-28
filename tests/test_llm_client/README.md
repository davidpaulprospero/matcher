# test_llm_client - LLM Client Tests

This directory contains tests for the unified LLM client abstraction that provides consistent interfaces across Gemini, Anthropic, and Ollama providers.

## Directory Overview

| File | Purpose |
|------|---------|
| `conftest.py` | Pytest fixtures for LLM client tests |
| `test_base.py` | Base classes (LLMRequest, LLMResponse, ResponseFormat, LLMClient) |
| `test_cache.py` | LLMCache TTL, key generation, stats, corruption handling |
| `test_factory.py` | `create_client()` factory, provider selection, API key handling |
| `test_parsers.py` | JSON parsing utilities (parse_json, parse_json_array, repair_json) |
| `test_retry.py` | Retry logic, exponential backoff, timeout detection |

## Key Fixtures

### temp_dir

Provides a temporary directory for file-based testing (e.g., cache files):

```python
@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)
```

### mock_api_key

Provides a mock API key for testing without real credentials:

```python
@pytest.fixture
def mock_api_key():
    """Provide a mock API key for testing."""
    return "test_api_key_12345"
```

### sample_json_response / sample_json_array_response

Provides sample JSON strings for testing parser functionality:

```python
@pytest.fixture
def sample_json_response():
    """Provide sample JSON response for testing."""
    return '{"status": "success", "data": {"id": 1, "name": "test"}}'

@pytest.fixture
def sample_json_array_response():
    """Provide sample JSON array response for testing."""
    return '[{"id": 1, "name": "first"}, {"id": 2, "name": "second"}]'
```

### cleanup_env_vars (autouse)

Auto-cleanup fixture that prevents environment variable pollution between tests:

```python
@pytest.fixture(autouse=True)
def cleanup_env_vars(monkeypatch):
    """Cleanup environment variables after each test."""
    yield
    # Cleanup is automatic with monkeypatch
```

## LLM Client Testing Patterns

### Mocking LLM Responses

There are three approaches to mock LLM responses, depending on test scope:

#### 1. Mock at Client Creation Level

Best for testing factory and client initialization:

```python
def test_create_gemini_client():
    """Test creating Gemini client."""
    client = create_client(
        provider="gemini",
        api_key="test_key",
        model="gemini-2.0-flash"
    )
    assert isinstance(client, GeminiClient)
    assert client.provider_name == "gemini"
```

#### 2. Mock the `_call_api` Method

Best for testing generate() logic without real API calls:

```python
class MockLLMClient(LLMClient):
    """Mock LLM client for testing base class functionality."""

    @property
    def provider_name(self) -> str:
        return "mock"

    def _call_api(self, request: LLMRequest) -> str:
        """Return mock response based on prompt."""
        if "json" in request.prompt.lower():
            return '{"result": "success"}'
        return "Mock response"

def test_generate_json():
    """Test generating and parsing JSON response."""
    client = MockLLMClient(api_key="test", model="test")
    request = LLMRequest(
        prompt="Return JSON",
        response_format=ResponseFormat.JSON,
        use_cache=False
    )
    response = client.generate(request)

    assert response.parsed_data == {"result": "success"}
```

#### 3. Mock with `unittest.mock.patch`

Best for integration tests or when you need to test existing client classes:

```python
from unittest.mock import Mock, patch

def test_with_mocked_api():
    """Test client with patched API calls."""
    with patch.object(GeminiClient, '_call_api') as mock_api:
        mock_api.return_value = '{"keywords": ["test", "example"]}'

        client = create_client("gemini", api_key="test")
        request = LLMRequest(
            prompt="Extract keywords",
            response_format=ResponseFormat.JSON,
            use_cache=False
        )
        response = client.generate(request)

        assert response.parsed_data == {"keywords": ["test", "example"]}
        mock_api.assert_called_once()
```

## Testing API Rate Limiting and Retry Logic

The retry module (`src/llm_client/retry.py`) implements exponential backoff with automatic error detection.

### Testing Successful Retry After Failures

```python
def test_retry_on_exception():
    """Test function retries on exception."""
    call_count = 0

    def func():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("Temporary error")
        return "success"

    result = with_retry(func, max_retries=3, base_delay=0.01)

    assert result == "success"
    assert call_count == 3
```

### Testing Max Retries Exhaustion

```python
def test_max_retries_exhausted():
    """Test that max retries are exhausted and error is raised."""
    call_count = 0

    def func():
        nonlocal call_count
        call_count += 1
        raise Exception("Persistent error")

    with pytest.raises(LLMProviderError) as exc_info:
        with_retry(func, max_retries=3, base_delay=0.01)

    assert call_count == 3
    assert "failed after 3 attempts" in str(exc_info.value).lower()
```

### Testing Timeout Detection

The retry logic automatically converts timeout-related errors to `LLMTimeoutError`:

```python
def test_timeout_error_converted():
    """Test that timeout errors are converted to LLMTimeoutError."""
    def func():
        raise Exception("Request timeout exceeded")

    with pytest.raises(LLMTimeoutError):
        with_retry(func, max_retries=2, base_delay=0.01)

def test_timeout_in_error_message():
    """Test timeout errors are detected correctly."""
    # Various timeout patterns
    for msg in ["Connection timeout after 30 seconds", "Deadline exceeded"]:
        with pytest.raises(LLMTimeoutError):
            with_retry(lambda: (_ for _ in ()).throw(Exception(msg)), max_retries=1, base_delay=0.01)
```

### Testing Exponential Backoff

```python
def test_exponential_backoff():
    """Test exponential backoff delays."""
    call_times = []

    def func():
        call_times.append(time.time())
        if len(call_times) < 3:
            raise Exception("Retry")
        return "success"

    with_retry(func, max_retries=3, base_delay=0.1)

    # First delay should be ~1.0s (0.1^0)
    delay1 = call_times[1] - call_times[0]
    assert 0.8 < delay1 < 1.2

    # Second delay should be ~0.1s (0.1^1)
    delay2 = call_times[2] - call_times[1]
    assert 0.05 < delay2 < 0.2
```

## Testing JSON Parsing (Validation)

The parsers module (`src/llm_client/parsers.py`) handles LLM response parsing with multiple fallback strategies.

### Testing JSON Extraction

```python
def test_json_with_markdown():
    """Test parsing JSON wrapped in markdown code blocks."""
    text = '```json\n{"key": "value"}\n```'
    result = parse_json(text)
    assert result == {"key": "value"}

def test_json_with_text_around():
    """Test extracting JSON from text with surrounding content."""
    text = 'Here is the result:\n{"key": "value"}\nThat was it.'
    result = parse_json(text)
    assert result == {"key": "value"}
```

### Testing JSON Repair

```python
def test_json_with_trailing_comma():
    """Test repairing JSON with trailing comma."""
    text = '{"key": "value",}'
    result = parse_json(text)
    assert result == {"key": "value"}

def test_fix_missing_comma_between_objects():
    """Test adding missing comma between objects."""
    text = '{"a": 1}{"b": 2}'
    repaired = repair_json(text)
    assert '},{'  in repaired
```

### Testing Array Parsing

```python
def test_valid_array():
    """Test parsing valid JSON array."""
    text = '[{"id": 1}, {"id": 2}, {"id": 3}]'
    result = parse_json_array(text)
    assert result == [{"id": 1}, {"id": 2}, {"id": 3}]

def test_extract_individual_objects():
    """Test extracting individual objects when array parsing fails."""
    text = '{"id": 1} {"id": 2} {"id": 3}'
    result = parse_json_array(text)
    assert len(result) >= 2  # Extracts objects even without array brackets
```

### Testing Key Extraction

```python
def test_extract_simple_values():
    """Test extracting values by keys from malformed JSON."""
    text = '"name": "John", "age": 30'
    result = extract_json_by_keys(text, ["name", "age"])
    assert result == {"name": "John", "age": 30}
```

## Testing Provider Selection (Gemini, Anthropic, Ollama)

### Testing Factory Provider Selection

```python
def test_create_gemini_client():
    client = create_client(provider="gemini", api_key="test_key")
    assert isinstance(client, GeminiClient)
    assert client.provider_name == "gemini"

def test_create_anthropic_client():
    client = create_client(provider="anthropic", api_key="test_key")
    assert isinstance(client, AnthropicClient)
    assert client.provider_name == "anthropic"

def test_create_ollama_client():
    client = create_client(provider="ollama", model="llama3.2")
    assert isinstance(client, OllamaClient)
    assert client.provider_name == "ollama"
```

### Testing Provider Aliases

```python
def test_create_gemini_with_google_alias():
    """Test creating Gemini client using 'google' alias."""
    client = create_client(provider="google", api_key="test_key")
    assert isinstance(client, GeminiClient)

def test_create_anthropic_with_claude_alias():
    """Test creating Anthropic client using 'claude' alias."""
    client = create_client(provider="claude", api_key="test_key")
    assert isinstance(client, AnthropicClient)
```

### Testing Case Insensitivity

```python
def test_case_insensitive_provider():
    """Test provider name is case-insensitive."""
    client1 = create_client(provider="GEMINI", api_key="test")
    client2 = create_client(provider="Gemini", api_key="test")
    client3 = create_client(provider="gemini", api_key="test")

    assert all(isinstance(c, GeminiClient) for c in [client1, client2, client3])
```

### Testing API Key Requirements

```python
def test_gemini_missing_api_key_raises_error(monkeypatch):
    """Test that missing Gemini API key raises error."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(LLMProviderError) as exc_info:
        create_client(provider="gemini")

    assert "API key required" in str(exc_info.value)

def test_ollama_no_api_key_required():
    """Test that Ollama doesn't require API key."""
    client = create_client(provider="ollama")  # Should not raise
    assert isinstance(client, OllamaClient)
```

### Testing Environment Variable Loading

```python
def test_gemini_api_key_from_env(monkeypatch):
    """Test Gemini API key loaded from environment."""
    monkeypatch.setenv("GEMINI_API_KEY", "env_key")
    client = create_client(provider="gemini")
    assert client.api_key == "env_key"

def test_explicit_api_key_overrides_env(monkeypatch):
    """Test explicit API key overrides environment variable."""
    monkeypatch.setenv("GEMINI_API_KEY", "env_key")
    client = create_client(provider="gemini", api_key="explicit_key")
    assert client.api_key == "explicit_key"
```

### Testing Unknown Provider Error

```python
def test_unknown_provider_raises_error():
    """Test that unknown provider raises error."""
    with pytest.raises(LLMProviderError) as exc_info:
        create_client(provider="unknown_provider", api_key="test")

    assert "Unknown provider" in str(exc_info.value)
    # Error message should list valid providers
    assert "gemini" in str(exc_info.value).lower()
    assert "anthropic" in str(exc_info.value).lower()
    assert "ollama" in str(exc_info.value).lower()
```

## Testing Streaming vs Non-Streaming Responses

The current LLM client implementation uses **non-streaming responses** exclusively. All providers are configured with `stream: False` to ensure complete response handling and caching compatibility.

### Why Non-Streaming

1. **Caching**: Responses are cached for efficiency; streaming would require buffering before caching
2. **JSON Parsing**: The `parse_json`/`parse_json_array` functions need complete text
3. **Retry Logic**: Retries are simpler with complete responses
4. **Pipeline Integration**: Downstream stages expect complete responses

### Testing Non-Streaming Behavior

```python
def test_complete_response_returned():
    """Test that generate() returns complete response, not iterator."""
    client = MockLLMClient(api_key="test", model="test")
    request = LLMRequest(prompt="Hello", use_cache=False)

    response = client.generate(request)

    # Response is a complete LLMResponse object
    assert isinstance(response, LLMResponse)
    assert isinstance(response.text, str)  # Not a generator
    assert response.parsed_data is None or isinstance(response.parsed_data, (dict, list))
```

### Testing Response Caching (Requires Non-Streaming)

```python
def test_cached_response_is_complete(temp_cache_dir):
    """Test cached responses are complete and can be re-used."""
    cache = LLMCache(temp_cache_dir, provider="test")

    request = LLMRequest(prompt="test prompt")
    response = LLMResponse(
        text="complete response text",
        parsed_data={"result": "success"},
        provider="test",
        model="test-model"
    )

    cache.set(request, response)
    cached = cache.get(request)

    # Cached data is complete, not partial
    assert cached["text"] == "complete response text"
    assert cached["parsed_data"] == {"result": "success"}
```

### If Streaming Were Needed

For streaming use cases (e.g., interactive UIs), you would need to:

1. Add `stream=True` parameter to `LLMRequest`
2. Implement streaming in provider `_call_api` methods
3. Bypass caching for streaming requests
4. Return generator/iterator instead of `LLMResponse`

Example test pattern for future streaming support:

```python
# Hypothetical future streaming test
def test_streaming_response():
    """Test streaming response yields chunks."""
    client = create_client("gemini", api_key="test")
    request = LLMRequest(prompt="Hello", stream=True)  # Future parameter

    response = client.generate_stream(request)  # Future method

    chunks = list(response)
    assert len(chunks) > 0
    full_text = "".join(chunks)
    assert len(full_text) > 0
```

## Testing LLM Cache

### Testing Cache Hit/Miss

```python
def test_set_and_get(temp_cache_dir):
    """Test setting and getting cache entries."""
    cache = LLMCache(temp_cache_dir, provider="test")

    request = LLMRequest(prompt="test prompt")
    response = LLMResponse(text="test response", provider="test")

    cache.set(request, response)
    cached = cache.get(request)

    assert cached is not None
    assert cached["text"] == "test response"

def test_cache_miss(temp_cache_dir):
    """Test cache returns None for miss."""
    cache = LLMCache(temp_cache_dir, provider="test")
    request = LLMRequest(prompt="nonexistent")

    cached = cache.get(request)
    assert cached is None
```

### Testing TTL Expiration

```python
def test_ttl_expiration(temp_cache_dir):
    """Test cache entries expire based on TTL."""
    cache = LLMCache(temp_cache_dir, provider="test", ttl_hours=1/3600)  # 1 second

    request = LLMRequest(prompt="test")
    response = LLMResponse(text="test")

    cache.set(request, response)
    assert cache.get(request) is not None

    time.sleep(1.5)  # Wait for expiration

    assert cache.get(request) is None  # Expired
```

### Testing Corrupt Cache Handling

```python
def test_corrupt_cache_file_handling(temp_cache_dir):
    """Test handling of corrupt cache files."""
    cache = LLMCache(temp_cache_dir, provider="test")

    request = LLMRequest(prompt="test")
    cache.set(request, LLMResponse(text="test"))

    # Corrupt the cache file
    cache_key = cache._cache_key(request)
    cache_file = cache.cache_dir / f"{request.cache_key_prefix}_{cache_key}.json"
    with open(cache_file, 'w') as f:
        f.write("not valid json{{{")

    # Should return None and delete corrupt file
    cached = cache.get(request)
    assert cached is None
    assert not cache_file.exists()
```

## Running Tests

```bash
# Run all LLM client tests
pytest tests/test_llm_client/ -v

# Run specific test file
pytest tests/test_llm_client/test_retry.py -v

# Run specific test class
pytest tests/test_llm_client/test_factory.py::TestCreateClient -v

# Run with markers
pytest tests/test_llm_client/ -m "fast" -v

# Run single test
pytest tests/test_llm_client/test_cache.py::TestLLMCache::test_ttl_expiration -v

# Run with coverage
pytest tests/test_llm_client/ --cov=src/llm_client --cov-report=term-missing
```

## Test Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.fast` | Quick unit tests (no I/O or delays) |
| `@pytest.mark.slow` | Tests with delays (e.g., TTL tests) |
| `@pytest.mark.requires_api` | Tests requiring real API credentials |

## Related Documentation

- `src/llm_client/__init__.py` - Module exports and usage examples
- `src/llm_client/base.py` - Base class documentation
- `CLAUDE.md` - Rule 9: LLM client usage guidelines
