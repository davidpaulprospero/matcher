"""
Performance benchmarks for embedding generation.

Run with: pytest tests/benchmarks/test_embedding_performance.py -v
"""

import pytest
import time
import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestEmbeddingPerformance:
    """Benchmark embedding generation operations."""

    @pytest.mark.slow
    @pytest.mark.requires_api
    @pytest.mark.flaky(reruns=2, reruns_delay=1.0)
    def test_voyage_embedding_speed(self, sample_segments):
        """Benchmark Voyage AI embedding generation speed."""
        pytest.skip("Requires VOYAGE_API_KEY - run manually")

        from src.embeddings import generate_embeddings_voyage
        import os

        if not os.getenv('VOYAGE_API_KEY'):
            pytest.skip("VOYAGE_API_KEY not set")

        # Test with small batch
        texts = [seg.text for seg in sample_segments[:10]]

        start = time.time()
        embeddings = generate_embeddings_voyage(texts)
        elapsed = time.time() - start

        print(f"\nVoyage embeddings: {elapsed:.2f}s for {len(texts)} texts ({elapsed/len(texts):.3f}s per text)")
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Throughput: {len(texts)/elapsed:.1f} texts/second")

        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] > 0  # Has embedding dimensions

    def test_embedding_batch_processing(self, sample_segments, benchmark):
        """Benchmark batch embedding processing."""
        # Mock embedding function for testing
        def mock_generate_embeddings(texts):
            # Simulate embedding generation
            return np.random.randn(len(texts), 1024).astype(np.float32)

        texts = [seg.text for seg in sample_segments]

        result = benchmark(mock_generate_embeddings, texts)

        assert result.shape[0] == len(texts)
        print(f"\nProcessed {len(texts)} embeddings, shape: {result.shape}")

    def test_embedding_similarity_computation(self, benchmark):
        """Benchmark cosine similarity computation speed."""
        from src.embeddings import cosine_similarity

        # Create mock embeddings
        query_embedding = np.random.randn(1024).astype(np.float32)
        candidate_embeddings = np.random.randn(1000, 1024).astype(np.float32)

        def compute_similarities():
            # Compute similarity for each candidate
            return [cosine_similarity(query_embedding, candidate) for candidate in candidate_embeddings]

        result = benchmark(compute_similarities)

        assert len(result) == 1000
        print(f"\nComputed {len(result)} similarities")

        # Calculate throughput
        comparisons_per_second = len(result) / benchmark.stats.stats.mean
        print(f"Similarity throughput: {comparisons_per_second:.0f} comparisons/second")

    def test_embedding_cache_performance(self, sample_segments, temp_benchmark_dir, benchmark):
        """Benchmark embedding cache write performance."""
        from src.embeddings import EmbeddingCache

        cache = EmbeddingCache(str(temp_benchmark_dir))
        embeddings = np.random.randn(len(sample_segments), 1024).astype(np.float32)
        test_key = "test_key"

        # Benchmark write (convert numpy to list for JSON serialization)
        def write_cache():
            cache.set(test_key, embeddings.tolist())
            return cache.get(test_key)  # Also test retrieval in same operation

        result = benchmark(write_cache)
        assert result is not None

        # Note: Cache stores as JSON (converted from numpy)
        print(f"\nCached {len(sample_segments)} embeddings with shape {embeddings.shape}")


class TestEmbeddingMemoryUsage:
    """Memory profiling for embedding operations."""

    def test_embedding_storage_memory(self, sample_segments):
        """Profile memory usage of embedding storage."""
        import tracemalloc

        num_embeddings = len(sample_segments)
        embedding_dim = 1024

        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[0]

        # Create embeddings
        embeddings = np.random.randn(num_embeddings, embedding_dim).astype(np.float32)

        peak_memory = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        memory_used_mb = (peak_memory - initial_memory) / 1024 / 1024
        expected_mb = (num_embeddings * embedding_dim * 4) / 1024 / 1024  # 4 bytes per float32

        print(f"\nEmbedding memory usage: {memory_used_mb:.2f} MB")
        print(f"Expected (theoretical): {expected_mb:.2f} MB")
        print(f"Overhead: {(memory_used_mb - expected_mb):.2f} MB")

        # Should be reasonable (Python has overhead for object management)
        # Allow 5x overhead for Python object structures and numpy array metadata
        assert memory_used_mb < expected_mb * 5

    def test_similarity_computation_memory(self):
        """Profile memory usage of similarity computation."""
        import tracemalloc

        query_embedding = np.random.randn(1024).astype(np.float32)
        candidate_embeddings = np.random.randn(10000, 1024).astype(np.float32)

        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[0]

        # Compute similarities
        from src.embeddings import cosine_similarity
        similarities = [cosine_similarity(query_embedding, candidate) for candidate in candidate_embeddings]

        peak_memory = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        memory_used_mb = (peak_memory - initial_memory) / 1024 / 1024
        print(f"\nSimilarity computation memory: {memory_used_mb:.2f} MB for 10,000 candidates")

        # Should use less than 100MB
        assert memory_used_mb < 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
