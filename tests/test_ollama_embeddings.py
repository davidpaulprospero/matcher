"""
Tests for Ollama embedding provider and related features.

Covers:
- OllamaEmbeddings class: init, availability check, embed with prefixes
- embed_mode parameter on all providers
- Provider-specific cache keys in compute_embeddings
- get_embedding_provider() ollama branch

All tests use mocked HTTP — no Ollama server needed.
"""

import pytest
import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, PropertyMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import (
    EmbeddingProvider,
    GeminiEmbeddings,
    VoyageEmbeddings,
    LocalEmbeddings,
    OllamaEmbeddings,
    EmbeddingCache,
    compute_embeddings,
    get_embedding_provider,
    _to_numpy,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for cache"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_requests():
    """Mock the requests module for OllamaEmbeddings."""
    with patch.dict('sys.modules', {}):
        mock_req = MagicMock()
        # Default: server available with model present
        tags_response = MagicMock()
        tags_response.json.return_value = {
            'models': [{'name': 'nomic-embed-text:latest'}]
        }
        mock_req.get.return_value = tags_response

        embed_response = MagicMock()
        embed_response.json.return_value = {
            'embeddings': [[0.1] * 768]
        }
        mock_req.post.return_value = embed_response

        # Wire up exception classes
        mock_req.ConnectionError = ConnectionError
        mock_req.Timeout = TimeoutError

        yield mock_req


@pytest.fixture
def ollama_provider(mock_requests):
    """Create an OllamaEmbeddings with mocked requests."""
    with patch('src.embeddings.OllamaEmbeddings.__init__', lambda self, *a, **kw: None):
        provider = OllamaEmbeddings.__new__(OllamaEmbeddings)
        provider._requests = mock_requests
        provider.model = "nomic-embed-text"
        provider.base_url = "http://localhost:11434"
        provider.timeout = 120
        return provider


# ============================================================================
# TestOllamaEmbeddings
# ============================================================================

class TestOllamaEmbeddings:
    """Tests for the OllamaEmbeddings class."""

    @pytest.mark.fast
    def test_init_calls_verify_availability(self, mock_requests):
        """Init should check server availability."""
        with patch.object(OllamaEmbeddings, '_verify_availability') as mock_verify:
            with patch('builtins.__import__', side_effect=lambda name, *a, **kw: mock_requests if name == 'requests' else __import__(name, *a, **kw)):
                # Use __new__ + manual init to test
                provider = OllamaEmbeddings.__new__(OllamaEmbeddings)
                provider._requests = mock_requests
                provider.model = "nomic-embed-text"
                provider.base_url = "http://localhost:11434"
                provider.timeout = 120
                provider._verify_availability()
                mock_verify.assert_called_once()

    @pytest.mark.fast
    def test_verify_server_not_running(self, mock_requests):
        """Should raise RuntimeError with 'ollama serve' hint when server is down."""
        mock_requests.get.side_effect = ConnectionError("Connection refused")

        provider = OllamaEmbeddings.__new__(OllamaEmbeddings)
        provider._requests = mock_requests
        provider.model = "nomic-embed-text"
        provider.base_url = "http://localhost:11434"
        provider.timeout = 120

        with pytest.raises(RuntimeError, match="ollama serve"):
            provider._verify_availability()

    @pytest.mark.fast
    def test_verify_model_not_pulled(self, mock_requests):
        """Should raise RuntimeError with 'ollama pull' hint when model missing."""
        tags_response = MagicMock()
        tags_response.json.return_value = {'models': [{'name': 'llama3:latest'}]}
        mock_requests.get.return_value = tags_response

        provider = OllamaEmbeddings.__new__(OllamaEmbeddings)
        provider._requests = mock_requests
        provider.model = "nomic-embed-text"
        provider.base_url = "http://localhost:11434"
        provider.timeout = 120

        with pytest.raises(RuntimeError, match="ollama pull"):
            provider._verify_availability()

    @pytest.mark.fast
    def test_verify_model_present(self, mock_requests):
        """Should not raise when model is available."""
        tags_response = MagicMock()
        tags_response.json.return_value = {
            'models': [
                {'name': 'nomic-embed-text:latest'},
                {'name': 'llama3:latest'}
            ]
        }
        mock_requests.get.return_value = tags_response

        provider = OllamaEmbeddings.__new__(OllamaEmbeddings)
        provider._requests = mock_requests
        provider.model = "nomic-embed-text"
        provider.base_url = "http://localhost:11434"
        provider.timeout = 120

        # Should not raise
        provider._verify_availability()

    @pytest.mark.fast
    def test_embed_document_prefix(self, ollama_provider, mock_requests):
        """Document mode should prepend 'search_document: ' prefix."""
        embed_response = MagicMock()
        embed_response.json.return_value = {'embeddings': [[0.1] * 768]}
        mock_requests.post.return_value = embed_response

        ollama_provider.embed(["test text"], embed_mode="document")

        call_args = mock_requests.post.call_args
        payload = call_args[1]['json'] if 'json' in call_args[1] else call_args[0][1] if len(call_args[0]) > 1 else None
        if payload is None:
            payload = call_args.kwargs.get('json')
        assert payload['input'] == ["search_document: test text"]

    @pytest.mark.fast
    def test_embed_query_prefix(self, ollama_provider, mock_requests):
        """Query mode should prepend 'search_query: ' prefix."""
        embed_response = MagicMock()
        embed_response.json.return_value = {'embeddings': [[0.2] * 768]}
        mock_requests.post.return_value = embed_response

        ollama_provider.embed(["find me something"], embed_mode="query")

        call_args = mock_requests.post.call_args
        payload = call_args.kwargs.get('json') or call_args[1].get('json')
        assert payload['input'] == ["search_query: find me something"]

    @pytest.mark.fast
    def test_embed_default_is_document(self, ollama_provider, mock_requests):
        """Default embed_mode should be 'document'."""
        embed_response = MagicMock()
        embed_response.json.return_value = {'embeddings': [[0.1] * 768]}
        mock_requests.post.return_value = embed_response

        ollama_provider.embed(["some text"])

        call_args = mock_requests.post.call_args
        payload = call_args.kwargs.get('json') or call_args[1].get('json')
        assert payload['input'][0].startswith("search_document: ")

    @pytest.mark.fast
    def test_embed_response_parsing(self, ollama_provider, mock_requests):
        """Should return embeddings list from response."""
        expected = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        embed_response = MagicMock()
        embed_response.json.return_value = {'embeddings': expected}
        mock_requests.post.return_value = embed_response

        result = ollama_provider.embed(["text1", "text2"])
        assert result == expected

    @pytest.mark.fast
    def test_embed_connection_error(self, ollama_provider, mock_requests):
        """Should raise RuntimeError on connection failure during embed."""
        mock_requests.post.side_effect = ConnectionError("Connection refused")

        with pytest.raises(RuntimeError, match="ollama serve"):
            ollama_provider.embed(["test"])

    @pytest.mark.fast
    def test_embed_timeout_error(self, ollama_provider, mock_requests):
        """Should raise RuntimeError on timeout during embed."""
        mock_requests.post.side_effect = TimeoutError("timed out")

        with pytest.raises(RuntimeError, match="timed out"):
            ollama_provider.embed(["test"])

    @pytest.mark.fast
    def test_embed_payload_structure(self, ollama_provider, mock_requests):
        """Payload should include model, keep_alive, and truncate."""
        embed_response = MagicMock()
        embed_response.json.return_value = {'embeddings': [[0.1] * 768]}
        mock_requests.post.return_value = embed_response

        ollama_provider.embed(["test"])

        call_args = mock_requests.post.call_args
        payload = call_args.kwargs.get('json') or call_args[1].get('json')
        assert payload['model'] == "nomic-embed-text"
        assert payload['keep_alive'] == "5m"
        assert payload['truncate'] is True

    @pytest.mark.fast
    def test_embed_empty_text_handling(self, ollama_provider, mock_requests):
        """Empty texts should be replaced with '[empty]'."""
        embed_response = MagicMock()
        embed_response.json.return_value = {'embeddings': [[0.1] * 768]}
        mock_requests.post.return_value = embed_response

        ollama_provider.embed(["", "  "])

        call_args = mock_requests.post.call_args
        payload = call_args.kwargs.get('json') or call_args[1].get('json')
        assert payload['input'] == ["search_document: [empty]", "search_document: [empty]"]

    @pytest.mark.fast
    def test_embed_posts_to_correct_url(self, ollama_provider, mock_requests):
        """Should POST to {base_url}/api/embed."""
        embed_response = MagicMock()
        embed_response.json.return_value = {'embeddings': [[0.1] * 768]}
        mock_requests.post.return_value = embed_response

        ollama_provider.embed(["test"])

        url = mock_requests.post.call_args[0][0]
        assert url == "http://localhost:11434/api/embed"


# ============================================================================
# TestEmbedModeInterface
# ============================================================================

class TestEmbedModeInterface:
    """Tests that embed_mode is correctly handled across providers."""

    @pytest.mark.fast
    def test_base_class_has_embed_mode_param(self):
        """EmbeddingProvider.embed() should accept embed_mode."""
        import inspect
        sig = inspect.signature(EmbeddingProvider.embed)
        assert 'embed_mode' in sig.parameters
        assert sig.parameters['embed_mode'].default == "document"

    @pytest.mark.fast
    def test_base_class_embed_batch_has_embed_mode(self):
        """EmbeddingProvider.embed_batch() should accept embed_mode."""
        import inspect
        sig = inspect.signature(EmbeddingProvider.embed_batch)
        assert 'embed_mode' in sig.parameters
        assert sig.parameters['embed_mode'].default == "document"

    @pytest.mark.fast
    def test_gemini_maps_query_to_retrieval_query(self):
        """Gemini should use retrieval_query task_type for query mode."""
        with patch('google.generativeai.configure'), \
             patch('google.generativeai.embed_content') as mock_embed:
            mock_embed.return_value = {'embedding': [[0.1] * 768]}

            with patch.dict('sys.modules', {'google.generativeai': MagicMock()}):
                provider = GeminiEmbeddings.__new__(GeminiEmbeddings)
                mock_genai = MagicMock()
                mock_genai.embed_content.return_value = {'embedding': [[0.1] * 768]}
                provider.genai = mock_genai
                provider.model = "models/gemini-embedding-001"

                provider.embed(["test"], embed_mode="query")
                call_kwargs = mock_genai.embed_content.call_args[1]
                assert call_kwargs['task_type'] == "retrieval_query"

    @pytest.mark.fast
    def test_gemini_maps_document_to_retrieval_document(self):
        """Gemini should use retrieval_document task_type for document mode."""
        provider = GeminiEmbeddings.__new__(GeminiEmbeddings)
        mock_genai = MagicMock()
        mock_genai.embed_content.return_value = {'embedding': [[0.1] * 768]}
        provider.genai = mock_genai
        provider.model = "models/gemini-embedding-001"

        provider.embed(["test"], embed_mode="document")
        call_kwargs = mock_genai.embed_content.call_args[1]
        assert call_kwargs['task_type'] == "retrieval_document"

    @pytest.mark.fast
    def test_embed_batch_forwards_mode(self):
        """embed_batch should forward embed_mode to embed()."""
        provider = EmbeddingProvider.__new__(EmbeddingProvider)
        provider.embed = MagicMock(return_value=[[0.1] * 768])

        provider.embed_batch(["test"], batch_size=100, embed_mode="query", max_retries=1)
        provider.embed.assert_called_once_with(["test"], embed_mode="query")


# ============================================================================
# TestProviderSpecificCacheKeys
# ============================================================================

class TestProviderSpecificCacheKeys:
    """Tests that cache keys include provider name."""

    def _make_provider(self, cls_name):
        """Create a mock provider whose type().__name__ returns cls_name."""
        # Create a real subclass so type(provider).__name__ works correctly
        cls = type(cls_name, (EmbeddingProvider,), {
            'embed': lambda self, texts, embed_mode="document": [[0.1] * 768] * len(texts)
        })
        return cls()

    @pytest.mark.fast
    def test_cache_key_includes_provider_name(self, temp_dir):
        """Cache key should be prefixed with provider class name."""
        provider = self._make_provider('GeminiEmbeddings')

        cache = MagicMock()
        cache.cache_dir = str(temp_dir)

        # Patch EmbeddingCache to capture the cache_key used
        with patch('src.embeddings.EmbeddingCache') as MockCache:
            mock_instance = MagicMock()
            mock_instance.get_cached_embeddings.return_value = ([], ["text1"], [0])
            MockCache.return_value = mock_instance

            compute_embeddings(
                texts=["text1"],
                provider=provider,
                cache=cache,
                cache_key="voiceover",
            )

            # Check that get_cached_embeddings was called with provider-qualified key
            call_args = mock_instance.get_cached_embeddings.call_args[0]
            assert call_args[1] == "gemini_voiceover"

    @pytest.mark.fast
    def test_different_providers_get_different_keys(self, temp_dir):
        """Different providers should not share cache keys."""
        cache = MagicMock()
        cache.cache_dir = str(temp_dir)

        keys_used = []

        for cls_name in ['GeminiEmbeddings', 'OllamaEmbeddings']:
            provider = self._make_provider(cls_name)

            with patch('src.embeddings.EmbeddingCache') as MockCache:
                mock_instance = MagicMock()
                mock_instance.get_cached_embeddings.return_value = ([], ["text1"], [0])
                MockCache.return_value = mock_instance

                compute_embeddings(
                    texts=["text1"],
                    provider=provider,
                    cache=cache,
                    cache_key="segments",
                )
                keys_used.append(mock_instance.get_cached_embeddings.call_args[0][1])

        assert keys_used[0] != keys_used[1]
        assert "gemini_" in keys_used[0]
        assert "ollama_" in keys_used[1]

    @pytest.mark.fast
    def test_same_provider_reuses_cache_key(self, temp_dir):
        """Same provider should get the same qualified cache key."""
        cache = MagicMock()
        cache.cache_dir = str(temp_dir)

        keys_used = []

        for _ in range(2):
            provider = self._make_provider('GeminiEmbeddings')

            with patch('src.embeddings.EmbeddingCache') as MockCache:
                mock_instance = MagicMock()
                mock_instance.get_cached_embeddings.return_value = ([], ["text1"], [0])
                MockCache.return_value = mock_instance

                compute_embeddings(
                    texts=["text1"],
                    provider=provider,
                    cache=cache,
                    cache_key="voiceover",
                )
                keys_used.append(mock_instance.get_cached_embeddings.call_args[0][1])

        assert keys_used[0] == keys_used[1]


# ============================================================================
# TestGetEmbeddingProvider
# ============================================================================

class TestGetEmbeddingProvider:
    """Tests for get_embedding_provider() ollama branch."""

    @pytest.mark.fast
    def test_ollama_provider_returned(self):
        """provider='ollama' should return OllamaEmbeddings."""
        config = MagicMock()
        config.embedding.provider = 'ollama'
        config.embedding.ollama_model = 'nomic-embed-text'
        config.embedding.ollama_base_url = 'http://localhost:11434'

        with patch.object(OllamaEmbeddings, '__init__', return_value=None) as mock_init:
            result = get_embedding_provider(config)
            assert isinstance(result, OllamaEmbeddings)
            mock_init.assert_called_once_with('nomic-embed-text', 'http://localhost:11434')

    @pytest.mark.fast
    def test_ollama_fallback_to_local_on_failure(self):
        """Should fall back to local if Ollama init fails."""
        config = MagicMock()
        config.embedding.provider = 'ollama'
        config.embedding.ollama_model = 'nomic-embed-text'
        config.embedding.ollama_base_url = 'http://localhost:11434'
        config.embedding.local_model = 'all-MiniLM-L6-v2'

        with patch.object(OllamaEmbeddings, '__init__', side_effect=RuntimeError("not running")), \
             patch.object(LocalEmbeddings, '__init__', return_value=None):
            result = get_embedding_provider(config)
            assert isinstance(result, LocalEmbeddings)

    @pytest.mark.fast
    def test_ollama_config_fields_passed_through(self):
        """Config fields should be passed to OllamaEmbeddings constructor."""
        config = MagicMock()
        config.embedding.provider = 'ollama'
        config.embedding.ollama_model = 'mxbai-embed-large'
        config.embedding.ollama_base_url = 'http://remote:11434'

        with patch.object(OllamaEmbeddings, '__init__', return_value=None) as mock_init:
            get_embedding_provider(config)
            mock_init.assert_called_once_with('mxbai-embed-large', 'http://remote:11434')


# ============================================================================
# TestComputeEmbeddingsEmbedMode
# ============================================================================

class TestComputeEmbeddingsEmbedMode:
    """Tests that embed_mode flows through compute_embeddings to provider."""

    @pytest.mark.fast
    def test_embed_mode_forwarded_to_provider(self, temp_dir):
        """compute_embeddings should pass embed_mode to provider.embed()."""
        provider = MagicMock(spec=EmbeddingProvider)
        provider.__class__ = type('GeminiEmbeddings', (), {})
        provider.embed = MagicMock(return_value=[[0.1] * 768])

        cache = MagicMock()
        cache.cache_dir = str(temp_dir)

        with patch('src.embeddings.EmbeddingCache') as MockCache:
            mock_instance = MagicMock()
            mock_instance.get_cached_embeddings.return_value = ([], ["hello"], [0])
            MockCache.return_value = mock_instance

            compute_embeddings(
                texts=["hello"],
                provider=provider,
                cache=cache,
                cache_key="test",
                embed_mode="query",
            )

            provider.embed.assert_called_once()
            call_kwargs = provider.embed.call_args[1]
            assert call_kwargs['embed_mode'] == "query"

    @pytest.mark.fast
    def test_embed_mode_default_is_document(self, temp_dir):
        """Default embed_mode should be 'document'."""
        provider = MagicMock(spec=EmbeddingProvider)
        provider.__class__ = type('GeminiEmbeddings', (), {})
        provider.embed = MagicMock(return_value=[[0.1] * 768])

        cache = MagicMock()
        cache.cache_dir = str(temp_dir)

        with patch('src.embeddings.EmbeddingCache') as MockCache:
            mock_instance = MagicMock()
            mock_instance.get_cached_embeddings.return_value = ([], ["hello"], [0])
            MockCache.return_value = mock_instance

            compute_embeddings(
                texts=["hello"],
                provider=provider,
                cache=cache,
                cache_key="test",
            )

            call_kwargs = provider.embed.call_args[1]
            assert call_kwargs['embed_mode'] == "document"
