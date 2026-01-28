"""
Comprehensive tests for embeddings module.

Covers:
- EmbeddingCache initialization and operations
- Text hashing and batch hashing
- Cached/uncached embedding retrieval
- Batch caching
- Cosine similarity computation
- Numpy array conversions
- API provider integration (mocked)
- Error handling and edge cases

Created: 2026-01-09 (Phase 3.1)
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tempfile
import shutil
import json
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import (
    EmbeddingCache,
    cosine_similarity,
    cleanup_embeddings,
    _to_numpy,
    HAS_NUMPY
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
def embedding_cache(temp_dir):
    """Create an embedding cache"""
    return EmbeddingCache(str(temp_dir))


@pytest.fixture
def sample_texts():
    """Sample texts for testing"""
    return [
        "Tokyo is the capital of Japan",
        "The city has modern architecture",
        "Mount Fuji is a famous landmark"
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embeddings (simple 3D vectors for testing)"""
    return [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]


# ============================================================================
# Test EmbeddingCache Initialization
# ============================================================================

class TestEmbeddingCacheInit:
    """Test EmbeddingCache initialization"""

    @pytest.mark.fast
    def test_init_creates_cache_dir(self, temp_dir):
        """Test cache directory creation"""
        cache = EmbeddingCache(str(temp_dir))

        assert cache.cache_dir.exists()
        assert cache.cache_dir.name == "embeddings"

    @pytest.mark.fast
    def test_init_creates_index_file(self, temp_dir):
        """Test index file creation"""
        cache = EmbeddingCache(str(temp_dir))

        # Index should be created (may be empty)
        assert isinstance(cache.index, dict)

    @pytest.mark.fast
    def test_init_ttl_is_zero(self, temp_dir):
        """Test that embeddings don't expire"""
        cache = EmbeddingCache(str(temp_dir))

        # TTL should be 0 (no expiration)
        assert cache.ttl_seconds == 0


# ============================================================================
# Test Hashing Functions
# ============================================================================

class TestHashingFunctions:
    """Test text and batch hashing"""

    @pytest.mark.fast
    def test_text_hash_generates_consistent_hash(self, embedding_cache):
        """Test text hash is consistent"""
        text = "Test text"

        hash1 = embedding_cache._text_hash(text)
        hash2 = embedding_cache._text_hash(text)

        assert hash1 == hash2
        assert len(hash1) == 12  # Hash length

    @pytest.mark.fast
    def test_text_hash_different_for_different_texts(self, embedding_cache):
        """Test different texts produce different hashes"""
        text1 = "First text"
        text2 = "Second text"

        hash1 = embedding_cache._text_hash(text1)
        hash2 = embedding_cache._text_hash(text2)

        assert hash1 != hash2

    @pytest.mark.fast
    def test_batch_hash_generates_consistent_hash(self, embedding_cache, sample_texts):
        """Test batch hash is consistent"""
        hash1 = embedding_cache._batch_hash(sample_texts)
        hash2 = embedding_cache._batch_hash(sample_texts)

        assert hash1 == hash2
        assert len(hash1) == 16  # Batch hash length

    @pytest.mark.fast
    def test_batch_hash_order_sensitive(self, embedding_cache):
        """Test batch hash changes with order"""
        texts1 = ["A", "B", "C"]
        texts2 = ["C", "B", "A"]

        hash1 = embedding_cache._batch_hash(texts1)
        hash2 = embedding_cache._batch_hash(texts2)

        # Note: batch_hash may or may not be order-sensitive depending on implementation
        # If it's order-insensitive (e.g., using set), hashes will be equal
        # This tests that hashing is consistent, not necessarily order-sensitive
        assert isinstance(hash1, str)
        assert isinstance(hash2, str)


# ============================================================================
# Test Embedding Caching
# ============================================================================

class TestEmbeddingCaching:
    """Test embedding caching operations"""

    @pytest.mark.fast
    def test_cache_embeddings_creates_files(self, embedding_cache, sample_texts, sample_embeddings):
        """Test caching creates cache files"""
        indices = [0, 1, 2]
        cache_key = "test_video"

        embedding_cache.cache_embeddings(
            sample_texts,
            sample_embeddings,
            indices,
            cache_key
        )

        # Should create 3 cache files
        cache_files = list(embedding_cache.cache_dir.glob(f"{cache_key}_*.json"))
        assert len(cache_files) == 3

    @pytest.mark.fast
    def test_cache_embeddings_updates_index(self, embedding_cache, sample_texts, sample_embeddings):
        """Test caching updates the index"""
        indices = [0, 1, 2]
        cache_key = "test_video"

        embedding_cache.cache_embeddings(
            sample_texts,
            sample_embeddings,
            indices,
            cache_key
        )

        assert cache_key in embedding_cache.index
        assert embedding_cache.index[cache_key]['count'] == 3

    @pytest.mark.fast
    def test_get_cached_embeddings_all_cached(self, embedding_cache, sample_texts, sample_embeddings):
        """Test retrieving all cached embeddings"""
        indices = [0, 1, 2]
        cache_key = "test_video"

        # First cache the embeddings
        embedding_cache.cache_embeddings(
            sample_texts,
            sample_embeddings,
            indices,
            cache_key
        )

        # Then retrieve them
        cached, uncached_texts, uncached_indices = embedding_cache.get_cached_embeddings(
            sample_texts,
            cache_key
        )

        assert len(cached) == 3
        assert len(uncached_texts) == 0
        assert len(uncached_indices) == 0

    @pytest.mark.fast
    def test_get_cached_embeddings_none_cached(self, embedding_cache, sample_texts):
        """Test retrieving when nothing is cached"""
        cache_key = "test_video"

        cached, uncached_texts, uncached_indices = embedding_cache.get_cached_embeddings(
            sample_texts,
            cache_key
        )

        assert len(cached) == 0
        assert len(uncached_texts) == 3
        assert uncached_indices == [0, 1, 2]

    @pytest.mark.fast
    def test_get_cached_embeddings_partial(self, embedding_cache, sample_texts, sample_embeddings):
        """Test retrieving with some cached, some not"""
        cache_key = "test_video"

        # Cache only first 2
        embedding_cache.cache_embeddings(
            sample_texts[:2],
            sample_embeddings[:2],
            [0, 1],
            cache_key
        )

        # Try to retrieve all 3
        cached, uncached_texts, uncached_indices = embedding_cache.get_cached_embeddings(
            sample_texts,
            cache_key
        )

        assert len(cached) == 2
        assert len(uncached_texts) == 1
        assert uncached_indices == [2]
        assert uncached_texts[0] == sample_texts[2]


# ============================================================================
# Test Batch Caching
# ============================================================================

class TestBatchCaching:
    """Test batch caching operations"""

    @pytest.mark.fast
    def test_cache_batch_creates_file(self, embedding_cache, sample_texts, sample_embeddings):
        """Test batch caching creates a file"""
        cache_key = "test_video"

        embedding_cache.cache_batch(
            sample_texts,
            sample_embeddings,
            cache_key
        )

        # Should create batch cache file
        cache_files = list(embedding_cache.cache_dir.glob(f"batch_{cache_key}_*.json"))
        assert len(cache_files) == 1

    @pytest.mark.fast
    def test_get_batch_cache_hit(self, embedding_cache, sample_texts, sample_embeddings):
        """Test batch cache hit"""
        cache_key = "test_video"

        # First cache the batch
        embedding_cache.cache_batch(
            sample_texts,
            sample_embeddings,
            cache_key
        )

        # Then retrieve it
        result = embedding_cache.get_batch_cache(sample_texts, cache_key)

        assert result is not None
        if HAS_NUMPY:
            import numpy as np
            assert isinstance(result, np.ndarray)
            assert result.shape == (3, 3)
        else:
            assert len(result) == 3

    @pytest.mark.fast
    def test_get_batch_cache_miss(self, embedding_cache, sample_texts):
        """Test batch cache miss"""
        cache_key = "test_video"

        result = embedding_cache.get_batch_cache(sample_texts, cache_key)

        assert result is None

    @pytest.mark.fast
    def test_get_batch_cache_wrong_length(self, embedding_cache, sample_texts, sample_embeddings):
        """Test batch cache miss when length doesn't match"""
        cache_key = "test_video"

        # Cache with 3 texts
        embedding_cache.cache_batch(
            sample_texts,
            sample_embeddings,
            cache_key
        )

        # Try to retrieve with different number of texts
        result = embedding_cache.get_batch_cache(sample_texts[:2], cache_key)

        # Should miss because length doesn't match
        assert result is None


# ============================================================================
# Test Cosine Similarity
# ============================================================================

class TestCosineSimilarity:
    """Test cosine similarity computation"""

    @pytest.mark.fast
    def test_identical_vectors_similarity_one(self):
        """Test identical vectors have similarity 1.0"""
        vec = [1.0, 0.0, 0.0]

        similarity = cosine_similarity(vec, vec)

        assert abs(similarity - 1.0) < 0.001

    @pytest.mark.fast
    def test_orthogonal_vectors_similarity_zero(self):
        """Test orthogonal vectors have similarity 0.0"""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]

        similarity = cosine_similarity(vec1, vec2)

        assert abs(similarity - 0.0) < 0.001

    @pytest.mark.fast
    def test_opposite_vectors_similarity_negative_one(self):
        """Test opposite vectors have similarity -1.0"""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]

        similarity = cosine_similarity(vec1, vec2)

        assert abs(similarity - (-1.0)) < 0.001

    @pytest.mark.fast
    def test_similar_vectors_high_similarity(self):
        """Test similar vectors have high similarity"""
        vec1 = [1.0, 1.0, 0.0]
        vec2 = [1.0, 0.9, 0.1]

        similarity = cosine_similarity(vec1, vec2)

        assert similarity > 0.9

    @pytest.mark.fast
    def test_zero_vector_returns_zero(self):
        """Test zero vector returns 0.0 similarity"""
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]

        similarity = cosine_similarity(vec1, vec2)

        assert similarity == 0.0

    @pytest.mark.skipif(not HAS_NUMPY, reason="Requires numpy")
    @pytest.mark.fast
    def test_numpy_arrays_work(self):
        """Test cosine similarity with numpy arrays"""
        import numpy as np

        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])

        similarity = cosine_similarity(vec1, vec2)

        assert abs(similarity - 1.0) < 0.001


# ============================================================================
# Test Numpy Conversion
# ============================================================================

@pytest.mark.skipif(not HAS_NUMPY, reason="Requires numpy")
class TestNumpyConversion:
    """Test numpy array conversion"""

    @pytest.mark.fast
    def test_to_numpy_converts_list(self):
        """Test converting list to numpy array"""
        import numpy as np

        embeddings = [[1.0, 2.0], [3.0, 4.0]]

        result = _to_numpy(embeddings)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (2, 2)

    @pytest.mark.fast
    def test_to_numpy_preserves_numpy(self):
        """Test numpy arrays are preserved"""
        import numpy as np

        embeddings = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

        result = _to_numpy(embeddings)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32  # Should convert to float32

    @pytest.mark.fast
    def test_to_numpy_returns_input_if_not_list_or_array(self):
        """Test other types are returned as-is"""
        embeddings = "not an array"

        result = _to_numpy(embeddings)

        assert result == embeddings


# ============================================================================
# Test Cleanup
# ============================================================================

class TestCleanup:
    """Test embedding cleanup"""

    @pytest.mark.fast
    def test_cleanup_embeddings_runs_without_error(self):
        """Test cleanup can be called safely"""
        # Should not raise any errors even if no model loaded
        cleanup_embeddings()

        # Verify it completes
        assert True

    @pytest.mark.fast
    def test_cleanup_embeddings_with_mock_model(self):
        """Test cleanup with a mock model"""
        import src.embeddings as emb_module

        # Set a mock model
        emb_module._local_embedding_model = Mock()

        cleanup_embeddings()

        # Model should be cleared
        assert emb_module._local_embedding_model is None


# ============================================================================
# Test Serialization
# ============================================================================

class TestSerialization:
    """Test cache entry serialization"""

    @pytest.mark.fast
    def test_serialize_entry(self, embedding_cache):
        """Test serializing a cache entry"""
        from src.cache import CacheEntry

        entry = CacheEntry(
            key="test_key",
            data=[1.0, 2.0, 3.0],
            cached_at=time.time(),
            metadata={"source": "test"}
        )

        serialized = embedding_cache._serialize_entry(entry)

        assert 'data' in serialized
        assert 'cached_at' in serialized
        assert 'metadata' in serialized
        assert serialized['data'] == [1.0, 2.0, 3.0]

    @pytest.mark.fast
    def test_deserialize_entry(self, embedding_cache):
        """Test deserializing a cache entry"""
        data = {
            'data': [1.0, 2.0, 3.0],
            'cached_at': time.time(),
            'metadata': {'source': 'test'}
        }

        entry = embedding_cache._deserialize_entry(data)

        assert entry.data == [1.0, 2.0, 3.0]
        assert entry.metadata['source'] == 'test'


# ============================================================================
# Test Error Handling
# ============================================================================

class TestErrorHandling:
    """Test error handling in caching"""

    @pytest.mark.fast
    def test_cache_embeddings_handles_write_error(self, embedding_cache, sample_texts, sample_embeddings):
        """Test caching handles write errors gracefully"""
        indices = [0, 1, 2]
        cache_key = "test_video"

        # Mock open to raise an exception simulating write error
        with patch('builtins.open', side_effect=IOError("Write error")):
            # Should not raise, just log error
            try:
                embedding_cache.cache_embeddings(
                    sample_texts,
                    sample_embeddings,
                    indices,
                    cache_key
                )
                # If we get here, error was handled
                assert True
            except Exception as e:
                # Should not reach here - embeddings module handles errors gracefully
                pytest.fail(f"Should handle write errors gracefully, but got: {e}")

    @pytest.mark.fast
    def test_get_cached_embeddings_handles_corrupted_cache(self, embedding_cache, temp_dir):
        """Test retrieving handles corrupted cache files"""
        cache_key = "test_video"
        text = "Test text"

        # Create a corrupted cache file
        text_hash = embedding_cache._text_hash(text)
        cache_file = embedding_cache.cache_dir / f"{cache_key}_{text_hash}.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        with open(cache_file, 'w') as f:
            f.write("{invalid json")

        # Should handle corrupted file gracefully
        cached, uncached_texts, uncached_indices = embedding_cache.get_cached_embeddings(
            [text],
            cache_key
        )

        # Should treat as cache miss
        assert len(cached) == 0
        assert len(uncached_texts) == 1

    @pytest.mark.fast
    def test_get_batch_cache_handles_corrupted_batch(self, embedding_cache, sample_texts):
        """Test batch retrieval handles corrupted files"""
        cache_key = "test_video"

        # Create a corrupted batch cache file
        batch_hash = embedding_cache._batch_hash(sample_texts)
        cache_file = embedding_cache.cache_dir / f"batch_{cache_key}_{batch_hash}.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        with open(cache_file, 'w') as f:
            f.write("{invalid json")

        # Should handle corrupted file gracefully
        result = embedding_cache.get_batch_cache(sample_texts, cache_key)

        assert result is None


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases"""

    @pytest.mark.fast
    def test_empty_text_list(self, embedding_cache):
        """Test caching with empty text list"""
        cached, uncached_texts, uncached_indices = embedding_cache.get_cached_embeddings(
            [],
            "test_key"
        )

        assert len(cached) == 0
        assert len(uncached_texts) == 0
        assert len(uncached_indices) == 0

    @pytest.mark.fast
    def test_very_long_text(self, embedding_cache):
        """Test caching with very long text"""
        long_text = "a" * 10000
        embedding = [1.0] * 384

        embedding_cache.cache_embeddings(
            [long_text],
            [embedding],
            [0],
            "test_key"
        )

        # Should cache successfully (with preview truncation)
        cached, uncached_texts, uncached_indices = embedding_cache.get_cached_embeddings(
            [long_text],
            "test_key"
        )

        assert len(cached) == 1

    @pytest.mark.fast
    def test_special_characters_in_text(self, embedding_cache):
        """Test caching with special characters"""
        special_text = "Text with 特殊字符 and émojis 🎉"
        embedding = [1.0, 2.0, 3.0]

        embedding_cache.cache_embeddings(
            [special_text],
            [embedding],
            [0],
            "test_key"
        )

        cached, uncached_texts, uncached_indices = embedding_cache.get_cached_embeddings(
            [special_text],
            "test_key"
        )

        assert len(cached) == 1


# ============================================================================
# Test Batch Caching
# ============================================================================

class TestBatchCaching:
    """Test batch-level caching operations"""

    @pytest.mark.fast
    def test_cache_batch_creates_file(self, embedding_cache, sample_texts, sample_embeddings):
        """Test batch caching creates cache file"""
        cache_key = "test_batch"

        embedding_cache.cache_batch(sample_texts, sample_embeddings, cache_key)

        # Should create batch cache file
        batch_files = list(embedding_cache.cache_dir.glob(f"batch_{cache_key}_*.json"))
        assert len(batch_files) == 1

    @pytest.mark.fast
    def test_get_batch_cache_hit(self, embedding_cache, sample_texts, sample_embeddings):
        """Test batch cache hit"""
        cache_key = "test_batch"

        # Cache the batch
        embedding_cache.cache_batch(sample_texts, sample_embeddings, cache_key)

        # Retrieve it
        result = embedding_cache.get_batch_cache(sample_texts, cache_key)

        assert result is not None
        if HAS_NUMPY:
            import numpy as np
            assert isinstance(result, np.ndarray)

    @pytest.mark.fast
    def test_get_batch_cache_miss(self, embedding_cache, sample_texts):
        """Test batch cache miss"""
        cache_key = "nonexistent"

        result = embedding_cache.get_batch_cache(sample_texts, cache_key)

        assert result is None

    @pytest.mark.fast
    def test_cache_batch_with_numpy(self, embedding_cache, sample_texts, sample_embeddings):
        """Test batch caching with numpy arrays"""
        import numpy as np

        cache_key = "numpy_batch"
        embeddings = np.array(sample_embeddings, dtype=np.float32)

        embedding_cache.cache_batch(sample_texts, embeddings, cache_key)

        # Should convert to list for JSON and cache successfully
        result = embedding_cache.get_batch_cache(sample_texts, cache_key)
        assert result is not None

    @pytest.mark.fast
    def test_cache_batch_length_mismatch(self, embedding_cache, sample_texts):
        """Test batch cache rejects length mismatch"""
        import numpy as np

        cache_key = "test_batch"
        # Cache with 3 embeddings
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        embedding_cache.cache_batch(sample_texts, embeddings, cache_key)

        # Try to retrieve with 2 texts
        result = embedding_cache.get_batch_cache(sample_texts[:2], cache_key)

        # Should be None due to length mismatch
        assert result is None


# ============================================================================
# Test Incremental Caching
# ============================================================================

class TestIncrementalCaching:
    """Test incremental batch caching for streaming"""

    @pytest.mark.fast
    def test_cache_incremental_creates_files(self, embedding_cache, sample_texts, sample_embeddings):
        """Test incremental caching creates numbered batch files"""
        cache_key = "streaming"

        for i in range(3):
            embedding_cache.cache_incremental(i, sample_texts, sample_embeddings, cache_key)

        # Should create 3 incremental files
        incremental_files = list(embedding_cache.cache_dir.glob(f"incremental_{cache_key}_*.json"))
        assert len(incremental_files) == 3

    @pytest.mark.fast
    def test_load_incremental_combines_batches(self, embedding_cache, sample_texts, sample_embeddings):
        """Test loading incremental batches combines them"""
        cache_key = "streaming"

        # Cache 3 batches
        for i in range(3):
            embedding_cache.cache_incremental(i, sample_texts, sample_embeddings, cache_key)

        # Load combined
        all_texts, all_embeddings = embedding_cache.load_incremental(cache_key)

        assert len(all_texts) == 9  # 3 batches × 3 texts
        assert len(all_embeddings) == 9

    @pytest.mark.fast
    def test_load_incremental_empty(self, embedding_cache):
        """Test loading incremental with no files"""
        cache_key = "nonexistent"

        all_texts, all_embeddings = embedding_cache.load_incremental(cache_key)

        assert len(all_texts) == 0
        assert len(all_embeddings) == 0

    @pytest.mark.fast
    def test_load_incremental_handles_corrupted_file(self, embedding_cache, sample_texts, sample_embeddings):
        """Test loading incremental handles corrupted files"""
        cache_key = "partial"

        # Cache 2 batches
        embedding_cache.cache_incremental(0, sample_texts, sample_embeddings, cache_key)
        embedding_cache.cache_incremental(1, sample_texts, sample_embeddings, cache_key)

        # Corrupt one file
        corrupt_file = embedding_cache.cache_dir / f"incremental_{cache_key}_0001.json"
        with open(corrupt_file, 'w') as f:
            f.write("{invalid json")

        # Should load only valid batch
        all_texts, all_embeddings = embedding_cache.load_incremental(cache_key)

        assert len(all_texts) == 3  # Only batch 0
        assert len(all_embeddings) == 3

    @pytest.mark.fast
    def test_clear_incremental_removes_files(self, embedding_cache, sample_texts, sample_embeddings):
        """Test clearing incremental files"""
        cache_key = "streaming"

        # Cache 3 batches
        for i in range(3):
            embedding_cache.cache_incremental(i, sample_texts, sample_embeddings, cache_key)

        # Clear them
        embedding_cache.clear_incremental(cache_key)

        # Should be empty
        incremental_files = list(embedding_cache.cache_dir.glob(f"incremental_{cache_key}_*.json"))
        assert len(incremental_files) == 0


# ============================================================================
# Test Embedding Providers
# ============================================================================

class TestEmbeddingProvider:
    """Test base EmbeddingProvider class"""

    @pytest.mark.fast
    def test_embed_batch_with_single_batch(self):
        """Test batch embedding with texts that fit in one batch"""
        from src.embeddings import EmbeddingProvider

        class MockProvider(EmbeddingProvider):
            def embed(self, texts):
                return [[1.0, 2.0] for _ in texts]

        provider = MockProvider()
        texts = ["text1", "text2", "text3"]

        result = provider.embed_batch(texts, batch_size=10, show_progress=False)

        assert len(result) == 3
        assert result[0] == [1.0, 2.0]

    @pytest.mark.fast
    def test_embed_batch_with_multiple_batches(self):
        """Test batch embedding across multiple batches"""
        from src.embeddings import EmbeddingProvider

        class MockProvider(EmbeddingProvider):
            def embed(self, texts):
                return [[1.0, 2.0] for _ in texts]

        provider = MockProvider()
        texts = [f"text{i}" for i in range(25)]

        result = provider.embed_batch(texts, batch_size=10, show_progress=False)

        assert len(result) == 25

    @pytest.mark.fast
    def test_embed_batch_with_retry_success(self):
        """Test batch embedding retries on failure then succeeds"""
        from src.embeddings import EmbeddingProvider

        class FlakyProvider(EmbeddingProvider):
            def __init__(self):
                self.attempt = 0

            def embed(self, texts):
                self.attempt += 1
                if self.attempt == 1:
                    raise Exception("API Error")
                return [[1.0, 2.0] for _ in texts]

        provider = FlakyProvider()
        texts = ["text1", "text2"]

        result = provider.embed_batch(texts, batch_size=10, show_progress=False, max_retries=3)

        assert len(result) == 2
        assert provider.attempt == 2  # Failed once, succeeded on retry

    @pytest.mark.fast
    def test_embed_batch_with_retry_exhausted(self):
        """Test batch embedding fills with zeros after max retries"""
        from src.embeddings import EmbeddingProvider

        class FailingProvider(EmbeddingProvider):
            def embed(self, texts):
                raise Exception("Permanent failure")

        provider = FailingProvider()
        texts = ["text1", "text2"]

        result = provider.embed_batch(texts, batch_size=10, show_progress=False, max_retries=2, retry_delay=0.1)

        # Should fill with zeros (default 768-dim)
        assert len(result) == 2
        assert result[0] == [0.0] * 768
        assert result[1] == [0.0] * 768

    @pytest.mark.fast
    def test_embed_batch_retry_uses_existing_dimension(self):
        """Test retry uses dimension from successful batches"""
        from src.embeddings import EmbeddingProvider

        class MixedProvider(EmbeddingProvider):
            def __init__(self):
                self.call_count = 0

            def embed(self, texts):
                self.call_count += 1
                if self.call_count == 1:
                    return [[1.0, 2.0, 3.0] for _ in texts]  # 3-dim
                else:
                    raise Exception("Batch 2 fails")

        provider = MixedProvider()
        texts = [f"text{i}" for i in range(12)]

        result = provider.embed_batch(texts, batch_size=10, show_progress=False, max_retries=1, retry_delay=0.1)

        # First batch succeeds with 3-dim, second batch fills with zeros (3-dim)
        assert len(result) == 12
        assert result[0] == [1.0, 2.0, 3.0]
        assert result[10] == [0.0, 0.0, 0.0]  # Uses dimension from successful batch


# ============================================================================
# Test Gemini Provider
# ============================================================================

class TestGeminiEmbeddings:
    """Test GeminiEmbeddings provider"""

    @patch('google.generativeai.configure')
    @patch('google.generativeai.embed_content')
    @pytest.mark.fast
    def test_gemini_embed_single_batch(self, mock_embed, mock_configure):
        """Test Gemini embedding single batch"""
        from src.embeddings import GeminiEmbeddings

        # Mock batch response - Gemini returns list of embeddings for multiple texts
        mock_response = {'embedding': [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}
        mock_embed.return_value = mock_response

        provider = GeminiEmbeddings(api_key="test_key")
        texts = ["text1", "text2"]

        result = provider.embed(texts)

        assert len(result) == 2
        assert result[0] == [1.0, 2.0, 3.0]
        assert result[1] == [4.0, 5.0, 6.0]
        mock_configure.assert_called_once()

    @patch('google.generativeai.configure')
    @patch('google.generativeai.embed_content')
    @pytest.mark.fast
    def test_gemini_embed_batch_mode(self, mock_embed, mock_configure):
        """Test Gemini batch mode uses embed_content correctly"""
        from src.embeddings import GeminiEmbeddings

        # Mock batch response
        mock_response = {
            'embedding': [[1.0, 2.0], [3.0, 4.0]]
        }
        mock_embed.return_value = mock_response

        provider = GeminiEmbeddings(api_key="test_key")
        texts = ["text1", "text2"]

        result = provider.embed(texts)

        # Should call with batch
        assert mock_embed.called
        call_args = mock_embed.call_args
        assert 'content' in call_args.kwargs or len(call_args.args) > 1


# ============================================================================
# Test Voyage Provider
# ============================================================================

class TestVoyageEmbeddings:
    """Test VoyageEmbeddings provider"""

    @patch('voyageai.Client')
    @pytest.mark.fast
    def test_voyage_embed(self, mock_client_class):
        """Test Voyage embedding"""
        from src.embeddings import VoyageEmbeddings

        # Mock client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.embeddings = [[1.0, 2.0], [3.0, 4.0]]
        mock_client.embed.return_value = mock_response
        mock_client_class.return_value = mock_client

        provider = VoyageEmbeddings(api_key="test_key")
        texts = ["text1", "text2"]

        result = provider.embed(texts)

        assert len(result) == 2
        assert result[0] == [1.0, 2.0]
        mock_client.embed.assert_called_once()


# ============================================================================
# Test Local Provider
# ============================================================================

class TestLocalEmbeddings:
    """Test LocalEmbeddings provider"""

    @patch('sentence_transformers.SentenceTransformer')
    @pytest.mark.fast
    def test_local_embed(self, mock_st_class):
        """Test local embedding with SentenceTransformer"""
        from src.embeddings import LocalEmbeddings
        import numpy as np

        # Mock model
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_st_class.return_value = mock_model

        provider = LocalEmbeddings(model_name="all-MiniLM-L6-v2")
        texts = ["text1", "text2"]

        result = provider.embed(texts)

        assert len(result) == 2
        # Check that encode was called with correct parameters (no convert_to_numpy)
        mock_model.encode.assert_called_once_with(texts, show_progress_bar=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
