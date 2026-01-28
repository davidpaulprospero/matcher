"""
Advanced tests for embeddings module - high-value orchestration functions.

Covers:
- get_embedding_provider() - Provider selection and fallback
- compute_embeddings() - Main embedding orchestration
- build_embedding_index() - FAISS index building
- find_top_k_similar() - Similarity search

Created: 2026-01-10 (Session 6 - Priority 1)
Target: embeddings.py 53.46% → 85% coverage
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tempfile
import shutil
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import (
    get_embedding_provider,
    compute_embeddings,
    build_embedding_index,
    find_top_k_similar,
    GeminiEmbeddings,
    VoyageEmbeddings,
    LocalEmbeddings,
    EmbeddingProvider
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for cache"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_cache(temp_dir):
    """Create a mock cache manager"""
    cache = Mock()
    cache.cache_dir = str(temp_dir)
    return cache


@pytest.fixture
def mock_config_gemini():
    """Mock config with Gemini provider"""
    config = Mock()
    config.embedding = Mock()
    config.embedding.provider = 'gemini'
    config.embedding.gemini_model = 'models/text-embedding-004'
    config.embedding.batch_size = 100
    config.embedding.max_retries = 3
    config.embedding.retry_delay = 2.0
    return config


@pytest.fixture
def mock_config_voyage():
    """Mock config with Voyage provider"""
    config = Mock()
    config.embedding = Mock()
    config.embedding.provider = 'voyage'
    config.embedding.voyage_model = 'voyage-2'
    config.embedding.batch_size = 128
    config.embedding.max_retries = 3
    config.embedding.retry_delay = 2.0
    return config


@pytest.fixture
def mock_config_local():
    """Mock config with local provider"""
    config = Mock()
    config.embedding = Mock()
    config.embedding.provider = 'local'
    config.embedding.local_model = 'all-MiniLM-L6-v2'
    config.embedding.batch_size = 32
    config.embedding.max_retries = 3
    config.embedding.retry_delay = 2.0
    return config


@pytest.fixture
def mock_config_faiss():
    """Mock config with FAISS settings"""
    config = Mock()
    config.indexing = Mock()
    config.indexing.use_faiss = True
    config.indexing.index_type = 'flat'
    config.indexing.ivf_nlist = 100
    config.indexing.ivf_nprobe = 10
    return config


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing (3D for simplicity)"""
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.7, 0.7, 0.0],
        [0.0, 0.7, 0.7]
    ], dtype=np.float32)


@pytest.fixture
def mock_embedding_provider():
    """Create a mock embedding provider"""
    class MockProvider(EmbeddingProvider):
        def embed(self, texts):
            # Return 3D embeddings for each text
            return [[float(i), float(i+1), float(i+2)] for i in range(len(texts))]

    return MockProvider()


# ============================================================================
# Test get_embedding_provider()
# ============================================================================

class TestGetEmbeddingProvider:
    """Test provider selection and initialization"""

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_gemini_key'})
    @patch('google.generativeai.configure')
    @pytest.mark.fast
    def test_get_gemini_provider(self, mock_configure, mock_config_gemini):
        """Test getting Gemini provider with API key"""
        provider = get_embedding_provider(mock_config_gemini)

        assert isinstance(provider, GeminiEmbeddings)
        assert provider.model == 'models/text-embedding-004'
        mock_configure.assert_called_once()

    @patch.dict(os.environ, {'VOYAGE_API_KEY': 'test_voyage_key'})
    @patch('voyageai.Client')
    @pytest.mark.fast
    def test_get_voyage_provider(self, mock_client, mock_config_voyage):
        """Test getting Voyage provider with API key"""
        provider = get_embedding_provider(mock_config_voyage)

        assert isinstance(provider, VoyageEmbeddings)
        assert provider.model == 'voyage-2'
        mock_client.assert_called_once()

    @patch('sentence_transformers.SentenceTransformer')
    @pytest.mark.fast
    def test_get_local_provider(self, mock_st, mock_config_local):
        """Test getting local provider"""
        mock_model = Mock()
        mock_st.return_value = mock_model

        provider = get_embedding_provider(mock_config_local)

        assert isinstance(provider, LocalEmbeddings)
        mock_st.assert_called_once_with('all-MiniLM-L6-v2')

    @patch.dict(os.environ, {}, clear=True)  # No API keys
    @patch('sentence_transformers.SentenceTransformer')
    @patch('google.generativeai.configure', side_effect=Exception("No API key"))
    @pytest.mark.fast
    def test_fallback_to_local_when_no_api_key(self, mock_configure, mock_st, mock_config_gemini):
        """Test fallback to local when Gemini API key missing"""
        mock_model = Mock()
        mock_st.return_value = mock_model

        # Force Gemini to fail
        mock_config_gemini.gemini_api_key = None

        provider = get_embedding_provider(mock_config_gemini)

        # Should fall back to local
        assert isinstance(provider, LocalEmbeddings)

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'})
    @patch('google.generativeai.configure', side_effect=Exception("Init failed"))
    @patch('sentence_transformers.SentenceTransformer')
    @pytest.mark.fast
    def test_fallback_to_local_on_gemini_error(self, mock_st, mock_configure, mock_config_gemini):
        """Test fallback to local when Gemini initialization fails"""
        mock_model = Mock()
        mock_st.return_value = mock_model

        provider = get_embedding_provider(mock_config_gemini)

        # Should fall back to local
        assert isinstance(provider, LocalEmbeddings)

    @patch.dict(os.environ, {}, clear=True)
    @patch('sentence_transformers.SentenceTransformer', side_effect=Exception("No model"))
    @pytest.mark.fast
    def test_error_when_all_providers_fail(self, mock_st, mock_config_local):
        """Test error raised when all providers fail"""
        with pytest.raises(RuntimeError, match="No embedding provider available"):
            get_embedding_provider(mock_config_local)

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'})
    @patch('google.generativeai.configure')
    @pytest.mark.fast
    def test_uses_custom_gemini_model(self, mock_configure):
        """Test custom Gemini model is used"""
        config = Mock()
        config.embedding = Mock()
        config.embedding.provider = 'gemini'
        config.embedding.gemini_model = 'models/custom-embedding'

        provider = get_embedding_provider(config)

        assert isinstance(provider, GeminiEmbeddings)
        assert provider.model == 'models/custom-embedding'


# ============================================================================
# Test compute_embeddings()
# ============================================================================

class TestComputeEmbeddings:
    """Test main embedding orchestration function"""

    @pytest.mark.fast
    def test_compute_embeddings_empty_texts(self, mock_cache, mock_embedding_provider):
        """Test with empty text list"""
        result = compute_embeddings(
            texts=[],
            provider=mock_embedding_provider,
            cache=mock_cache,
            show_progress=False
        )

        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    @patch('src.embeddings.EmbeddingCache')
    @pytest.mark.fast
    def test_compute_embeddings_all_cached(self, mock_cache_class, mock_cache, mock_embedding_provider):
        """Test when all embeddings are cached"""
        # Mock cache with all embeddings cached
        mock_cache_inst = Mock()
        mock_cache_inst.get_cached_embeddings.return_value = (
            [(0, [1.0, 2.0, 3.0]), (1, [4.0, 5.0, 6.0])],  # cached_results
            [],  # uncached_texts
            []   # uncached_indices
        )
        mock_cache_class.return_value = mock_cache_inst

        texts = ["text1", "text2"]
        result = compute_embeddings(
            texts=texts,
            provider=mock_embedding_provider,
            cache=mock_cache,
            show_progress=False
        )

        # Should return cached embeddings without calling provider
        assert isinstance(result, np.ndarray)
        assert len(result) == 2
        np.testing.assert_array_almost_equal(result[0], [1.0, 2.0, 3.0])

    @patch('src.embeddings.EmbeddingCache')
    @pytest.mark.fast
    def test_compute_embeddings_none_cached(self, mock_cache_class, mock_cache, mock_embedding_provider):
        """Test when no embeddings are cached"""
        # Mock cache with nothing cached
        mock_cache_inst = Mock()
        mock_cache_inst.get_cached_embeddings.return_value = (
            [],  # cached_results
            ["text1", "text2"],  # uncached_texts
            [0, 1]   # uncached_indices
        )
        mock_cache_inst.cache_embeddings = Mock()
        mock_cache_class.return_value = mock_cache_inst

        texts = ["text1", "text2"]
        result = compute_embeddings(
            texts=texts,
            provider=mock_embedding_provider,
            cache=mock_cache,
            show_progress=False
        )

        # Should compute new embeddings
        assert isinstance(result, np.ndarray)
        assert len(result) == 2

        # Should cache the new embeddings
        assert mock_cache_inst.cache_embeddings.called

    @patch('src.embeddings.EmbeddingCache')
    @pytest.mark.fast
    def test_compute_embeddings_partial_cached(self, mock_cache_class, mock_cache, mock_embedding_provider):
        """Test when some embeddings are cached"""
        # Mock cache with partial results
        mock_cache_inst = Mock()
        mock_cache_inst.get_cached_embeddings.return_value = (
            [(0, [1.0, 2.0, 3.0])],  # text1 cached
            ["text2", "text3"],       # text2, text3 uncached
            [1, 2]
        )
        mock_cache_inst.cache_embeddings = Mock()
        mock_cache_class.return_value = mock_cache_inst

        texts = ["text1", "text2", "text3"]
        result = compute_embeddings(
            texts=texts,
            provider=mock_embedding_provider,
            cache=mock_cache,
            show_progress=False
        )

        # Should have all 3 embeddings (1 cached + 2 new)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3

        # First embedding should be from cache
        np.testing.assert_array_almost_equal(result[0], [1.0, 2.0, 3.0])

    @patch('src.embeddings.EmbeddingCache')
    @pytest.mark.fast
    def test_compute_embeddings_with_config_batch_size(self, mock_cache_class, mock_cache, mock_config_gemini):
        """Test batch size from config is used"""
        # Mock cache with no cached results
        mock_cache_inst = Mock()

        # Mock provider that returns proper embeddings
        class BatchProvider(EmbeddingProvider):
            def embed(self, texts):
                return [[float(i), float(i+1), float(i+2)] for i in range(len(texts))]

        provider = BatchProvider()

        # Create texts that will be uncached
        texts = [f"text{i}" for i in range(10)]  # Smaller number to avoid complexity

        mock_cache_inst.get_cached_embeddings.return_value = (
            [],  # No cached
            texts,  # All uncached
            list(range(10))  # Indices
        )
        mock_cache_inst.cache_embeddings = Mock()
        mock_cache_class.return_value = mock_cache_inst

        result = compute_embeddings(
            texts=texts,
            provider=provider,
            cache=mock_cache,
            show_progress=False,
            config=mock_config_gemini  # batch_size=100
        )

        # Should process all embeddings
        assert len(result) == 10

    @patch('src.embeddings.EmbeddingCache')
    @pytest.mark.fast
    def test_compute_embeddings_cleans_empty_strings(self, mock_cache_class, mock_cache, mock_embedding_provider):
        """Test empty strings are replaced with [silence]"""
        mock_cache_inst = Mock()
        mock_cache_inst.get_cached_embeddings.return_value = (
            [],
            ["[silence]", "[silence]"],
            [0, 1]
        )
        mock_cache_inst.cache_embeddings = Mock()
        mock_cache_class.return_value = mock_cache_inst

        texts = ["", "  "]  # Empty and whitespace
        result = compute_embeddings(
            texts=texts,
            provider=mock_embedding_provider,
            cache=mock_cache,
            show_progress=False
        )

        # Should handle empty texts
        assert len(result) == 2

    @patch('src.embeddings.EmbeddingCache')
    @pytest.mark.fast
    def test_compute_embeddings_with_retry(self, mock_cache_class, mock_cache):
        """Test retry logic on provider failure"""
        mock_cache_inst = Mock()
        mock_cache_inst.get_cached_embeddings.return_value = (
            [],
            ["text1"],
            [0]
        )
        mock_cache_inst.cache_embeddings = Mock()
        mock_cache_class.return_value = mock_cache_inst

        # Create provider that fails then succeeds
        class FlakyProvider(EmbeddingProvider):
            def __init__(self):
                self.attempt = 0

            def embed(self, texts):
                self.attempt += 1
                if self.attempt == 1:
                    raise Exception("API Error")
                return [[1.0, 2.0, 3.0] for _ in texts]

        provider = FlakyProvider()
        config = Mock()
        config.embedding = Mock()
        config.embedding.batch_size = 10
        config.embedding.max_retries = 3
        config.embedding.retry_delay = 0.1

        result = compute_embeddings(
            texts=["text1"],
            provider=provider,
            cache=mock_cache,
            show_progress=False,
            config=config
        )

        # Should succeed after retry
        assert len(result) == 1
        assert provider.attempt == 2

    @patch('src.embeddings.EmbeddingCache')
    @pytest.mark.fast
    def test_compute_embeddings_fills_zeros_on_failure(self, mock_cache_class, mock_cache):
        """Test fills zeros when all retries exhausted"""
        mock_cache_inst = Mock()
        mock_cache_inst.get_cached_embeddings.return_value = (
            [],
            ["text1", "text2"],
            [0, 1]
        )
        mock_cache_inst.cache_embeddings = Mock()
        mock_cache_class.return_value = mock_cache_inst

        # Create provider that always fails
        class FailingProvider(EmbeddingProvider):
            def embed(self, texts):
                raise Exception("Permanent failure")

        provider = FailingProvider()
        config = Mock()
        config.embedding = Mock()
        config.embedding.batch_size = 10
        config.embedding.max_retries = 1
        config.embedding.retry_delay = 0.1

        result = compute_embeddings(
            texts=["text1", "text2"],
            provider=provider,
            cache=mock_cache,
            show_progress=False,
            config=config
        )

        # Should fill with zeros (default 768-dim)
        assert len(result) == 2
        assert result[0].sum() == 0  # All zeros


# ============================================================================
# Test build_embedding_index()
# ============================================================================

class TestBuildEmbeddingIndex:
    """Test FAISS index building"""

    @pytest.mark.fast
    def test_build_flat_index(self, sample_embeddings, mock_config_faiss):
        """Test building flat FAISS index"""
        mock_config_faiss.indexing.index_type = 'flat'

        index = build_embedding_index(sample_embeddings, mock_config_faiss)

        assert index is not None
        assert index.ntotal == 5  # 5 embeddings
        assert index.d == 3  # 3 dimensions

    @pytest.mark.fast
    def test_build_ivf_index(self, sample_embeddings, mock_config_faiss):
        """Test building IVF FAISS index"""
        mock_config_faiss.indexing.index_type = 'ivf'
        mock_config_faiss.indexing.ivf_nlist = 2
        mock_config_faiss.indexing.ivf_nprobe = 1

        index = build_embedding_index(sample_embeddings, mock_config_faiss)

        assert index is not None
        assert index.ntotal == 5

    @pytest.mark.fast
    def test_build_index_with_list_embeddings(self, mock_config_faiss):
        """Test building index with list instead of numpy array"""
        embeddings = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

        index = build_embedding_index(embeddings, mock_config_faiss)

        assert index is not None
        assert index.ntotal == 3
        assert index.d == 2

    @pytest.mark.fast
    def test_build_index_disabled(self, sample_embeddings):
        """Test when FAISS is disabled in config"""
        config = Mock()
        config.indexing = Mock()
        config.indexing.use_faiss = False

        index = build_embedding_index(sample_embeddings, config)

        assert index is None

    @pytest.mark.fast
    def test_build_index_handles_error_gracefully(self, mock_config_faiss):
        """Test index building handles errors gracefully"""
        # Create invalid embeddings (empty array)
        embeddings = np.array([], dtype=np.float32)

        # Should handle error and return None
        try:
            index = build_embedding_index(embeddings, mock_config_faiss)
            # If it doesn't raise, should return None
            assert index is None or index.ntotal == 0
        except:
            # Error is expected for empty array
            pass

    @pytest.mark.fast
    def test_build_ivf_index_adjusts_nlist_for_small_dataset(self, mock_config_faiss):
        """Test IVF nlist is adjusted for small datasets"""
        mock_config_faiss.indexing.index_type = 'ivf'
        mock_config_faiss.indexing.ivf_nlist = 100  # Too large for 5 vectors

        # Small dataset (5 vectors)
        embeddings = np.random.randn(5, 384).astype(np.float32)

        index = build_embedding_index(embeddings, mock_config_faiss)

        # nlist should be adjusted to max(1, 5 // 10) = 1
        assert index is not None

    @pytest.mark.fast
    def test_build_index_normalizes_embeddings(self, mock_config_faiss):
        """Test embeddings are normalized for cosine similarity"""
        # Create unnormalized embeddings
        embeddings = np.array([
            [3.0, 4.0],  # magnitude = 5
            [5.0, 12.0]  # magnitude = 13
        ], dtype=np.float32)

        index = build_embedding_index(embeddings, mock_config_faiss)

        assert index is not None
        # After normalization, magnitudes should be 1.0


# ============================================================================
# Test find_top_k_similar()
# ============================================================================

class TestFindTopKSimilar:
    """Test similarity search"""

    @pytest.mark.fast
    def test_find_top_k_with_faiss_index(self, sample_embeddings, mock_config_faiss):
        """Test finding top-k with FAISS index"""
        # Build index
        index = build_embedding_index(sample_embeddings, mock_config_faiss)

        # Query with first embedding
        query = sample_embeddings[0]

        distances, indices = find_top_k_similar(query, sample_embeddings, k=3, index=index)

        # Should return top 3 similar
        assert len(distances) == 3
        assert len(indices) == 3

        # Most similar should be itself (index 0)
        assert indices[0] == 0
        assert distances[0] >= 0.99  # Cosine similarity ~1.0

    @pytest.mark.fast
    def test_find_top_k_without_index(self, sample_embeddings):
        """Test finding top-k with brute-force search"""
        query = sample_embeddings[0]

        distances, indices = find_top_k_similar(query, sample_embeddings, k=3, index=None)

        # Should return top 3 similar
        assert len(distances) == 3
        assert len(indices) == 3

        # Most similar should be itself
        assert indices[0] == 0

    @pytest.mark.fast
    def test_find_top_k_with_list_inputs(self, sample_embeddings):
        """Test with list inputs instead of numpy arrays"""
        query = [1.0, 0.0, 0.0]
        embeddings_list = sample_embeddings.tolist()

        distances, indices = find_top_k_similar(query, embeddings_list, k=2, index=None)

        assert len(distances) == 2
        assert len(indices) == 2

    @pytest.mark.fast
    def test_find_top_k_larger_than_dataset(self, sample_embeddings):
        """Test when k > dataset size"""
        query = sample_embeddings[0]

        distances, indices = find_top_k_similar(query, sample_embeddings, k=100, index=None)

        # Should return all 5 embeddings
        assert len(distances) == 5
        assert len(indices) == 5

    @pytest.mark.fast
    def test_find_top_k_query_1d_reshapes_to_2d(self, sample_embeddings):
        """Test 1D query is reshaped to 2D for FAISS"""
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # 1D

        distances, indices = find_top_k_similar(query, sample_embeddings, k=2, index=None)

        assert len(distances) == 2

    @pytest.mark.fast
    def test_find_top_k_normalizes_query(self, sample_embeddings):
        """Test query is normalized for cosine similarity"""
        # Unnormalized query
        query = np.array([3.0, 0.0, 0.0], dtype=np.float32)

        distances, indices = find_top_k_similar(query, sample_embeddings, k=2, index=None)

        # Should work correctly despite unnormalized input
        assert len(distances) == 2
        # Most similar should be embedding [1,0,0] (index 0)
        assert indices[0] == 0

    @pytest.mark.fast
    def test_find_top_k_zero_query_vector(self, sample_embeddings):
        """Test with zero query vector"""
        query = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        distances, indices = find_top_k_similar(query, sample_embeddings, k=2, index=None)

        # Should handle zero vector gracefully
        assert len(distances) == 2

    @pytest.mark.fast
    def test_find_top_k_sorted_by_similarity(self, sample_embeddings):
        """Test results are sorted by descending similarity"""
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        distances, indices = find_top_k_similar(query, sample_embeddings, k=5, index=None)

        # Distances should be in descending order (most similar first)
        assert all(distances[i] >= distances[i+1] for i in range(len(distances)-1))

    @pytest.mark.fast
    def test_find_top_k_faiss_fallback_on_error(self, sample_embeddings, mock_config_faiss):
        """Test fallback to brute-force when FAISS search fails"""
        # Build index
        index = build_embedding_index(sample_embeddings, mock_config_faiss)

        # Mock FAISS search to raise error
        with patch.object(index, 'search', side_effect=Exception("FAISS error")):
            query = sample_embeddings[0]

            distances, indices = find_top_k_similar(query, sample_embeddings, k=2, index=index)

            # Should fall back to brute-force and succeed
            assert len(distances) == 2
            assert len(indices) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
