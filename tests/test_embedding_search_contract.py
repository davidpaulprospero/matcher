"""
Contract validation tests for EmbeddingSearch.

Validates the contract between pipeline callers and EmbeddingSearch:
- from_matching_config() accepts embedding_index=None (match stage passes None)
- search() returns valid candidates with None index (brute-force fallback)
- Graceful handling of empty video_embeddings
- Graceful handling of mismatched video_segments and video_embeddings lengths
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.matching.embedding_search import EmbeddingSearch, EmbeddingSearchConfig


class MockSRTSegment:
    """Mock segment for testing."""
    def __init__(self, segment_id: str, text: str = "test"):
        self.id = segment_id
        self.text = text


class TestFromMatchingConfigNoneIndex:
    """Contract: from_matching_config() must accept embedding_index=None."""

    def test_from_matching_config_accepts_none_index_explicitly(self):
        """Match stage passes embedding_index=None explicitly - must not raise."""
        mock_config = Mock()
        mock_config.embedding_candidates = 20
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        segments = [MockSRTSegment("seg0"), MockSRTSegment("seg1")]

        searcher = EmbeddingSearch.from_matching_config(
            mock_config, embeddings, segments, embedding_index=None
        )

        assert searcher.embedding_index is None
        assert searcher.video_embeddings == embeddings
        assert len(searcher.video_segments) == 2

    def test_from_matching_config_defaults_index_to_none(self):
        """When embedding_index is omitted, it defaults to None."""
        mock_config = Mock()
        mock_config.embedding_candidates = 20
        embeddings = [[0.1, 0.2]]
        segments = [MockSRTSegment("seg0")]

        searcher = EmbeddingSearch.from_matching_config(
            mock_config, embeddings, segments
        )

        assert searcher.embedding_index is None

    def test_from_matching_config_none_index_returns_functional_instance(self):
        """Instance created with None index must be usable for search."""
        mock_config = Mock()
        mock_config.embedding_candidates = 20
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.7, 0.7, 0.0]]
        segments = [MockSRTSegment(f"seg{i}") for i in range(3)]

        searcher = EmbeddingSearch.from_matching_config(
            mock_config, embeddings, segments, embedding_index=None
        )

        # Must not raise - brute-force path should work
        results = searcher.search([1.0, 0.0, 0.0])
        assert isinstance(results, list)
        assert len(results) == 3


class TestSearchBruteForceWithNoneIndex:
    """Contract: search() must return valid candidates when embedding_index is None."""

    def test_search_returns_candidates_without_faiss_index(self):
        """Brute-force fallback must return ranked candidates."""
        config = EmbeddingSearchConfig(embedding_candidates=20)
        embeddings = [
            [1.0, 0.0, 0.0],   # Most similar to query
            [0.0, 1.0, 0.0],   # Orthogonal
            [0.7, 0.7, 0.0],   # Partial similarity
        ]
        segments = [MockSRTSegment(f"seg{i}") for i in range(3)]

        searcher = EmbeddingSearch(config, embeddings, segments, embedding_index=None)
        results = searcher.search([1.0, 0.0, 0.0])

        assert len(results) == 3
        # Each result is (segment, score) tuple
        for seg, score in results:
            assert hasattr(seg, 'id')
            assert isinstance(score, float)

    def test_brute_force_results_are_sorted_by_similarity(self):
        """Brute-force results should be sorted descending by similarity."""
        config = EmbeddingSearchConfig(embedding_candidates=20)
        embeddings = [
            [1.0, 0.0, 0.0],   # Identical to query -> highest
            [0.0, 1.0, 0.0],   # Orthogonal -> lowest
            [0.7, 0.7, 0.0],   # Partial -> middle
        ]
        segments = [MockSRTSegment(f"seg{i}") for i in range(3)]

        searcher = EmbeddingSearch(config, embeddings, segments, embedding_index=None)
        results = searcher.search([1.0, 0.0, 0.0])

        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True), "Results must be sorted by descending similarity"

    def test_brute_force_highest_similarity_is_identical_vector(self):
        """Identical vector should have highest similarity score (~1.0)."""
        config = EmbeddingSearchConfig(embedding_candidates=20)
        embeddings = [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],   # Identical to query
            [0.5, 0.5, 0.0],
        ]
        segments = [MockSRTSegment(f"seg{i}") for i in range(3)]

        searcher = EmbeddingSearch(config, embeddings, segments, embedding_index=None)
        results = searcher.search([1.0, 0.0, 0.0])

        # First result should be seg1 (identical vector) with score ~1.0
        assert results[0][0].id == "seg1"
        assert results[0][1] == pytest.approx(1.0, abs=0.01)

    def test_search_with_none_index_produces_valid_segment_references(self):
        """All returned segments must be valid references from the input list."""
        config = EmbeddingSearchConfig(embedding_candidates=20)
        embeddings = [[float(i), float(i + 1), float(i + 2)] for i in range(5)]
        segments = [MockSRTSegment(f"seg{i}") for i in range(5)]

        searcher = EmbeddingSearch(config, embeddings, segments, embedding_index=None)
        results = searcher.search([1.0, 2.0, 3.0])

        returned_ids = {seg.id for seg, _ in results}
        valid_ids = {f"seg{i}" for i in range(5)}
        assert returned_ids.issubset(valid_ids)


class TestEmptyVideoEmbeddings:
    """Contract: EmbeddingSearch must handle empty video_embeddings gracefully.

    Note: find_top_k_similar in embeddings.py does not guard against empty arrays
    (numpy AxisError on empty input). These tests mock find_top_k_similar to validate
    that EmbeddingSearch itself handles empty results correctly. The underlying
    embeddings.py empty-array bug is a separate concern.
    """

    @patch('src.matching.embedding_search.find_top_k_similar')
    def test_search_with_empty_embeddings_returns_empty_list(self, mock_find_similar):
        """search() with no video embeddings must return empty list, not crash."""
        mock_find_similar.return_value = (np.array([]), np.array([], dtype=int))

        config = EmbeddingSearchConfig(embedding_candidates=20)
        searcher = EmbeddingSearch(config, [], [], embedding_index=None)
        results = searcher.search([1.0, 0.0, 0.0])

        assert results == []

    @patch('src.matching.embedding_search.find_top_k_similar')
    def test_from_matching_config_with_empty_embeddings(self, mock_find_similar):
        """from_matching_config() with empty embeddings must create valid instance."""
        mock_find_similar.return_value = (np.array([]), np.array([], dtype=int))
        mock_config = Mock()
        mock_config.embedding_candidates = 20

        searcher = EmbeddingSearch.from_matching_config(
            mock_config, [], [], embedding_index=None
        )

        assert searcher.video_embeddings == []
        assert searcher.video_segments == []
        # search should still work
        results = searcher.search([0.5, 0.5])
        assert results == []

    @patch('src.matching.embedding_search.find_top_k_similar')
    def test_empty_embeddings_with_faiss_index_none(self, mock_find_similar):
        """Empty embeddings + None index must not raise."""
        mock_find_similar.return_value = (np.array([]), np.array([], dtype=int))

        config = EmbeddingSearchConfig()
        searcher = EmbeddingSearch(config, [], [], embedding_index=None)

        # Should not raise any exception
        results = searcher.search([0.1, 0.2, 0.3])
        assert isinstance(results, list)
        assert len(results) == 0


class TestMismatchedSegmentsAndEmbeddings:
    """Contract: EmbeddingSearch must handle mismatched lengths gracefully."""

    @patch('src.matching.embedding_search.find_top_k_similar')
    def test_more_segments_than_embeddings_filters_invalid(self, mock_find_similar):
        """When segments > embeddings, indices beyond embedding count should be filtered."""
        # find_top_k_similar returns indices based on embeddings length
        # Indices 0,1 are valid for 2 embeddings, but we have 5 segments
        mock_find_similar.return_value = (
            np.array([0.9, 0.8]),
            np.array([0, 1])
        )

        config = EmbeddingSearchConfig()
        embeddings = [[0.1, 0.2], [0.3, 0.4]]  # 2 embeddings
        segments = [MockSRTSegment(f"seg{i}") for i in range(5)]  # 5 segments

        searcher = EmbeddingSearch(config, embeddings, segments, embedding_index=None)
        results = searcher.search([0.1, 0.2])

        # Indices 0 and 1 are valid (within segments range)
        assert len(results) == 2
        assert results[0][0].id == "seg0"
        assert results[1][0].id == "seg1"

    @patch('src.matching.embedding_search.find_top_k_similar')
    def test_more_embeddings_than_segments_filters_invalid(self, mock_find_similar):
        """When embeddings > segments, indices beyond segments count should be filtered."""
        # find_top_k_similar returns indices based on embeddings array
        # Index 3 would be valid for embeddings but out of range for segments
        mock_find_similar.return_value = (
            np.array([0.9, 0.85, 0.8]),
            np.array([0, 1, 3])  # Index 3 is out of range for 2 segments
        )

        config = EmbeddingSearchConfig()
        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]  # 4 embeddings
        segments = [MockSRTSegment("seg0"), MockSRTSegment("seg1")]  # Only 2 segments

        searcher = EmbeddingSearch(config, embeddings, segments, embedding_index=None)
        results = searcher.search([0.1, 0.2])

        # Only indices 0 and 1 are valid (within segments range)
        assert len(results) == 2
        assert results[0][0].id == "seg0"
        assert results[1][0].id == "seg1"

    @patch('src.matching.embedding_search.find_top_k_similar')
    def test_mismatched_does_not_raise(self, mock_find_similar):
        """Mismatched lengths must not raise exceptions."""
        mock_find_similar.return_value = (
            np.array([0.9]),
            np.array([0])
        )

        config = EmbeddingSearchConfig()
        # 3 embeddings, 1 segment
        embeddings = [[0.1], [0.2], [0.3]]
        segments = [MockSRTSegment("only_seg")]

        searcher = EmbeddingSearch(config, embeddings, segments, embedding_index=None)

        # Should not raise
        results = searcher.search([0.1])
        assert len(results) == 1
        assert results[0][0].id == "only_seg"

    @patch('src.matching.embedding_search.find_top_k_similar')
    def test_all_indices_out_of_segment_range_returns_empty(self, mock_find_similar):
        """When all returned indices exceed segment count, result should be empty."""
        mock_find_similar.return_value = (
            np.array([0.9, 0.8, 0.7]),
            np.array([5, 6, 7])  # All out of range for 2 segments
        )

        config = EmbeddingSearchConfig()
        embeddings = [[0.1] * 3] * 8  # 8 embeddings
        segments = [MockSRTSegment("seg0"), MockSRTSegment("seg1")]  # Only 2 segments

        searcher = EmbeddingSearch(config, embeddings, segments, embedding_index=None)
        results = searcher.search([0.1, 0.1, 0.1])

        assert results == []
