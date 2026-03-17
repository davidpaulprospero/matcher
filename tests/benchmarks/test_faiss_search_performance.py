"""
Performance benchmarks for FAISS search operations.

Run with: pytest tests/benchmarks/test_faiss_search_performance.py -v
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from typing import List, Tuple, Any
from unittest.mock import Mock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestFAISSSearchPerformance:
    """Benchmark FAISS index search operations."""

    @pytest.mark.fast
    def test_faiss_index_creation(self, benchmark):
        """Benchmark creating a FAISS index with embeddings."""
        # Check if FAISS is available
        try:
            import faiss
        except ImportError:
            pytest.skip("FAISS not installed")

        # Create random embeddings (simulate 10000 video segments)
        dim = 1024
        n_vectors = 10000
        embeddings = np.random.randn(n_vectors, dim).astype('float32')

        def create_index():
            index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
            index.add(embeddings)
            return index

        result = benchmark(create_index)
        assert result.ntotal == n_vectors
        print(f"\nCreated FAISS index with {result.ntotal} vectors")

    @pytest.mark.fast
    def test_faiss_search_small(self, benchmark):
        """Benchmark FAISS search with small index (1000 vectors)."""
        try:
            import faiss
        except ImportError:
            pytest.skip("FAISS not installed")

        dim = 1024
        n_vectors = 1000
        embeddings = np.random.randn(n_vectors, dim).astype('float32')
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # Single query
        query = np.random.randn(dim).astype('float32')

        def search():
            distances, indices = index.search(query.reshape(1, -1), k=20)
            return distances, indices

        result = benchmark(search)
        assert result[0].shape == (1, 20)
        print(f"\nSearched {n_vectors} vectors, returned {len(result[0][0])} results")

    @pytest.mark.fast
    def test_faiss_search_medium(self, benchmark):
        """Benchmark FAISS search with medium index (10000 vectors)."""
        try:
            import faiss
        except ImportError:
            pytest.skip("FAISS not installed")

        dim = 1024
        n_vectors = 10000
        embeddings = np.random.randn(n_vectors, dim).astype('float32')
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        query = np.random.randn(dim).astype('float32')

        def search():
            distances, indices = index.search(query.reshape(1, -1), k=20)
            return distances, indices

        result = benchmark(search)
        assert result[0].shape == (1, 20)
        print(f"\nSearched {n_vectors} vectors, returned {len(result[0][0])} results")

    @pytest.mark.fast
    def test_faiss_search_large(self, benchmark):
        """Benchmark FAISS search with large index (50000 vectors)."""
        try:
            import faiss
        except ImportError:
            pytest.skip("FAISS not installed")

        dim = 1024
        n_vectors = 50000
        embeddings = np.random.randn(n_vectors, dim).astype('float32')
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        query = np.random.randn(dim).astype('float32')

        def search():
            distances, indices = index.search(query.reshape(1, -1), k=20)
            return distances, indices

        result = benchmark(search)
        assert result[0].shape == (1, 20)
        print(f"\nSearched {n_vectors} vectors, returned {len(result[0][0])} results")

    @pytest.mark.fast
    def test_faiss_batch_search(self, benchmark):
        """Benchmark FAISS batch search (multiple queries)."""
        try:
            import faiss
        except ImportError:
            pytest.skip("FAISS not installed")

        dim = 1024
        n_vectors = 10000
        embeddings = np.random.randn(n_vectors, dim).astype('float32')
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # Batch of 50 queries
        queries = np.random.randn(50, dim).astype('float32')

        def batch_search():
            distances, indices = index.search(queries, k=20)
            return distances, indices

        result = benchmark(batch_search)
        assert result[0].shape == (50, 20)
        print(f"\nBatch searched {n_vectors} vectors with 50 queries")

    @pytest.mark.fast
    def test_faiss_ivf_search(self, benchmark):
        """Benchmark FAISS IVF index search for large datasets."""
        try:
            import faiss
        except ImportError:
            pytest.skip("FAISS not installed")

        dim = 512  # Lower dimension for faster index
        n_vectors = 50000
        embeddings = np.random.randn(n_vectors, dim).astype('float32')

        # Create IVF index
        nlist = 100  # Number of clusters
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)

        query = np.random.randn(dim).astype('float32')

        def search():
            distances, indices = index.search(query.reshape(1, -1), k=20)
            return distances, indices

        result = benchmark(search)
        assert result[0].shape == (1, 20)
        print(f"\nIVF search on {n_vectors} vectors, nlist={nlist}")


class TestFAISSIndexOptimization:
    """Benchmark FAISS index optimization techniques."""

    @pytest.mark.fast
    def test_faiss_normalization_impact(self, benchmark):
        """Benchmark impact of normalization on search quality."""
        try:
            import faiss
        except ImportError:
            pytest.skip("FAISS not installed")

        dim = 1024
        n_vectors = 5000
        embeddings = np.random.randn(n_vectors, dim).astype('float32')

        # With normalization
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        index = faiss.IndexFlatIP(dim)
        index.add(embeddings_norm)

        query = np.random.randn(dim).astype('float32')
        query_norm = query / np.linalg.norm(query)

        def search_normalized():
            distances, indices = index.search(query_norm.reshape(1, -1), k=10)
            return distances, indices

        result = benchmark(search_normalized)
        print(f"\nNormalized search completed")

    @pytest.mark.fast
    def test_faiss_k_selection_scaling(self, benchmark):
        """Benchmark how k selection affects search performance."""
        try:
            import faiss
        except ImportError:
            pytest.skip("FAISS not installed")

        dim = 1024
        n_vectors = 10000
        embeddings = np.random.randn(n_vectors, dim).astype('float32')
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        query = np.random.randn(dim).astype('float32')
        k_values = [5, 10, 20, 50, 100]

        def search_variable_k():
            results = []
            for k in k_values:
                distances, indices = index.search(query.reshape(1, -1), k=k)
                results.append((distances, indices))
            return results

        result = benchmark(search_variable_k)
        print(f"\nVariable k search completed for k values: {k_values}")


class TestEmbeddingSearchIntegration:
    """Benchmark integration with EmbeddingSearch class."""

    @pytest.mark.fast
    def test_embedding_search_with_faiss(self, benchmark):
        """Benchmark EmbeddingSearch using FAISS."""
        try:
            import faiss
        except ImportError:
            pytest.skip("FAISS not installed")

        from src.matching.embedding_search import EmbeddingSearch, EmbeddingSearchConfig

        dim = 512
        n_segments = 1000

        # Create mock video segments and embeddings
        video_embeddings = np.random.randn(n_segments, dim).astype('float32')
        video_embeddings = video_embeddings / np.linalg.norm(video_embeddings, axis=1, keepdims=True)

        video_segments = [
            Mock(index=i, start_time=i*5.0, end_time=(i+1)*5.0,
                 text=f"Video segment {i}", source_file=f"video_{i//10}.mp4")
            for i in range(n_segments)
        ]

        # Create FAISS index
        index = faiss.IndexFlatIP(dim)
        index.add(video_embeddings)

        config = EmbeddingSearchConfig(embedding_candidates=20)
        search = EmbeddingSearch(config, video_embeddings, video_segments, index)

        query = np.random.randn(dim).astype('float32')
        query = query / np.linalg.norm(query)

        def search_with_faiss():
            return search.search(query.tolist())

        result = benchmark(search_with_faiss)
        print(f"\nEmbeddingSearch with FAISS returned {len(result)} candidates")


class TestFAISSMemoryUsage:
    """Memory profiling for FAISS operations."""

    @pytest.mark.fast
    def test_faiss_index_memory(self):
        """Profile memory usage of FAISS index."""
        try:
            import faiss
        except ImportError:
            pytest.skip("FAISS not installed")

        import tracemalloc

        dim = 1024
        n_vectors = 50000

        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[0]

        embeddings = np.random.randn(n_vectors, dim).astype('float32')
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        peak_memory = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        memory_used_mb = (peak_memory - initial_memory) / 1024 / 1024
        expected_mb = (n_vectors * dim * 4) / 1024 / 1024  # float32 = 4 bytes

        print(f"\nFAISS index memory: {memory_used_mb:.2f} MB")
        print(f"Expected (theoretical): {expected_mb:.2f} MB")

        # Allow 2x overhead for FAISS internal structures
        assert memory_used_mb < expected_mb * 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
