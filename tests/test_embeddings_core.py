"""
Unit tests for embeddings core functions.

Tests cosine similarity computation which is a critical function.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import cosine_similarity


class TestCosineSimilarity:
    """Test cosine similarity computation."""

    def test_identical_vectors(self):
        """Test similarity of identical vectors is 1.0."""
        vec1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        vec2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        similarity = cosine_similarity(vec1, vec2)

        assert np.isclose(similarity, 1.0, atol=1e-6)

    def test_orthogonal_vectors(self):
        """Test similarity of orthogonal vectors is 0.0."""
        vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        similarity = cosine_similarity(vec1, vec2)

        assert np.isclose(similarity, 0.0, atol=1e-6)

    def test_opposite_vectors(self):
        """Test similarity of opposite vectors is -1.0."""
        vec1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        vec2 = np.array([-1.0, -2.0, -3.0], dtype=np.float32)

        similarity = cosine_similarity(vec1, vec2)

        assert np.isclose(similarity, -1.0, atol=1e-6)

    def test_similar_vectors(self):
        """Test similarity of similar vectors."""
        vec1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        vec2 = np.array([1.1, 2.1, 2.9], dtype=np.float32)

        similarity = cosine_similarity(vec1, vec2)

        assert 0.95 < similarity < 1.0

    def test_high_dimensional_vectors(self):
        """Test similarity with high-dimensional vectors (typical embedding size)."""
        vec1 = np.random.randn(1024).astype(np.float32)
        vec2 = vec1 + np.random.randn(1024).astype(np.float32) * 0.1

        similarity = cosine_similarity(vec1, vec2)

        assert -1.0 <= similarity <= 1.0

    def test_normalized_vectors(self):
        """Test similarity with already normalized vectors."""
        vec1 = np.array([0.6, 0.8, 0.0], dtype=np.float32)  # Magnitude = 1.0
        vec2 = np.array([0.8, 0.6, 0.0], dtype=np.float32)  # Magnitude = 1.0

        similarity = cosine_similarity(vec1, vec2)

        assert 0.0 <= similarity <= 1.0

    def test_zero_vector(self):
        """Test similarity with zero vector."""
        vec1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        vec2 = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Should handle gracefully (return 0.0 or raise)
        try:
            similarity = cosine_similarity(vec1, vec2)
            # If it doesn't raise, check it's a valid value
            assert -1.0 <= similarity <= 1.0
        except (ValueError, ZeroDivisionError):
            # Expected behavior for zero vector
            pass

    def test_performance(self):
        """Test cosine similarity is fast enough."""
        import time

        vec1 = np.random.randn(1024).astype(np.float32)
        vec2 = np.random.randn(1024).astype(np.float32)

        start = time.time()
        for _ in range(1000):
            cosine_similarity(vec1, vec2)
        elapsed = time.time() - start

        # Should compute 1000 similarities in under 100ms
        assert elapsed < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
