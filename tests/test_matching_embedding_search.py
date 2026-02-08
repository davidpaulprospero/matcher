"""
Tests for EmbeddingSearch class extracted from main.py.

Tests embedding-based similarity search functionality:
- Config parameter handling
- Top-k retrieval with mock embeddings
- Edge cases (empty index)
- Similarity calculation with known values
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.matching.embedding_search import EmbeddingSearch, EmbeddingSearchConfig


class MockSRTSegment:
    """Mock segment for testing."""
    def __init__(self, segment_id: str, text: str = "test"):
        self.id = segment_id
        self.text = text


class TestEmbeddingSearchInit:
    """Tests for EmbeddingSearch initialization and config handling."""

    def test_init_with_default_config(self):
        """Test initialization with default EmbeddingSearchConfig."""
        config = EmbeddingSearchConfig()
        embeddings = [[0.1, 0.2, 0.3]]
        segments = [MockSRTSegment("seg1")]

        searcher = EmbeddingSearch(config, embeddings, segments)

        assert searcher.config == config
        assert searcher.config.embedding_candidates == 20
        assert searcher.video_embeddings == embeddings
        assert searcher.video_segments == segments
        assert searcher.embedding_index is None

    def test_init_with_custom_candidates(self):
        """Test initialization with custom embedding_candidates count."""
        config = EmbeddingSearchConfig(embedding_candidates=50)
        embeddings = [[0.1, 0.2]]
        segments = [MockSRTSegment("seg1")]

        searcher = EmbeddingSearch(config, embeddings, segments)

        assert searcher.config.embedding_candidates == 50

    def test_init_with_embedding_index(self):
        """Test initialization with optional FAISS index."""
        config = EmbeddingSearchConfig()
        embeddings = [[0.1, 0.2]]
        segments = [MockSRTSegment("seg1")]
        mock_index = Mock()

        searcher = EmbeddingSearch(config, embeddings, segments, embedding_index=mock_index)

        assert searcher.embedding_index is mock_index

    def test_from_matching_config_extracts_candidates(self):
        """Test factory method extracts embedding_candidates from matching config."""
        mock_matching_config = Mock()
        mock_matching_config.embedding_candidates = 35
        embeddings = [[0.5, 0.5]]
        segments = [MockSRTSegment("seg1")]

        searcher = EmbeddingSearch.from_matching_config(
            mock_matching_config, embeddings, segments
        )

        assert searcher.config.embedding_candidates == 35

    def test_from_matching_config_uses_default_when_missing(self):
        """Test factory method uses default when config lacks embedding_candidates."""
        mock_matching_config = Mock(spec=[])  # No attributes
        embeddings = [[0.5, 0.5]]
        segments = [MockSRTSegment("seg1")]

        searcher = EmbeddingSearch.from_matching_config(
            mock_matching_config, embeddings, segments
        )

        assert searcher.config.embedding_candidates == 20


class TestEmbeddingSearchTopK:
    """Tests for top-k retrieval with mock embeddings."""

    @patch('src.matching.embedding_search.find_top_k_similar')
    def test_search_returns_top_k_candidates(self, mock_find_similar):
        """Test search returns correct number of top-k candidates."""
        # Setup mock return - 3 results with distances and indices
        mock_find_similar.return_value = (
            np.array([0.95, 0.80, 0.65]),  # distances (similarities)
            np.array([0, 2, 1])  # indices
        )

        config = EmbeddingSearchConfig(embedding_candidates=20)
        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        segments = [
            MockSRTSegment("seg0"),
            MockSRTSegment("seg1"),
            MockSRTSegment("seg2"),
        ]

        searcher = EmbeddingSearch(config, embeddings, segments)
        query = [0.1, 0.2]
        results = searcher.search(query)

        assert len(results) == 3
        # Results should be (segment, similarity) tuples in order from mock
        assert results[0][0].id == "seg0"
        assert results[0][1] == 0.95
        assert results[1][0].id == "seg2"
        assert results[1][1] == 0.80
        assert results[2][0].id == "seg1"
        assert results[2][1] == 0.65

    @patch('src.matching.embedding_search.find_top_k_similar')
    def test_search_respects_num_candidates_override(self, mock_find_similar):
        """Test search respects num_candidates parameter override."""
        mock_find_similar.return_value = (np.array([0.9]), np.array([0]))

        config = EmbeddingSearchConfig(embedding_candidates=20)
        searcher = EmbeddingSearch(config, [[0.1]], [MockSRTSegment("seg0")])

        searcher.search([0.1], num_candidates=50)

        # Should use 50 (override) not 20 (config default)
        call_args = mock_find_similar.call_args
        assert call_args[0][2] == 50  # k parameter is third positional arg

    @patch('src.matching.embedding_search.find_top_k_similar')
    def test_search_enforces_minimum_candidates(self, mock_find_similar):
        """Test search enforces minimum of 20 candidates for variety."""
        mock_find_similar.return_value = (np.array([0.9]), np.array([0]))

        config = EmbeddingSearchConfig(embedding_candidates=5)  # Too low
        searcher = EmbeddingSearch(config, [[0.1]], [MockSRTSegment("seg0")])

        searcher.search([0.1])

        # Should use 20 (minimum) not 5 (config value)
        call_args = mock_find_similar.call_args
        assert call_args[0][2] == 20  # k enforced to minimum


class TestEmbeddingSearchEmptyIndex:
    """Tests for edge case handling with empty index."""

    @patch('src.matching.embedding_search.find_top_k_similar')
    def test_handles_empty_embeddings_list(self, mock_find_similar):
        """Test search handles empty embeddings list gracefully."""
        mock_find_similar.return_value = (np.array([]), np.array([]))

        config = EmbeddingSearchConfig()
        searcher = EmbeddingSearch(config, [], [])

        results = searcher.search([0.1, 0.2])

        assert results == []

    @patch('src.matching.embedding_search.find_top_k_similar')
    def test_handles_no_valid_indices(self, mock_find_similar):
        """Test search filters out invalid indices."""
        # Return indices that are out of bounds
        mock_find_similar.return_value = (
            np.array([0.9, 0.8]),
            np.array([5, 10])  # Both out of bounds for 3 segments
        )

        config = EmbeddingSearchConfig()
        segments = [MockSRTSegment(f"seg{i}") for i in range(3)]
        searcher = EmbeddingSearch(config, [[0.1], [0.2], [0.3]], segments)

        results = searcher.search([0.1])

        # Should filter out invalid indices
        assert results == []

    @patch('src.matching.embedding_search.find_top_k_similar')
    def test_filters_negative_indices(self, mock_find_similar):
        """Test search filters out negative indices (FAISS failure case)."""
        mock_find_similar.return_value = (
            np.array([0.9, 0.8, 0.7]),
            np.array([-1, 0, 1])  # -1 indicates no match in FAISS
        )

        config = EmbeddingSearchConfig()
        segments = [MockSRTSegment("seg0"), MockSRTSegment("seg1")]
        searcher = EmbeddingSearch(config, [[0.1], [0.2]], segments)

        results = searcher.search([0.1])

        # Should only include valid indices (0 and 1, not -1)
        assert len(results) == 2
        assert results[0][0].id == "seg0"
        assert results[1][0].id == "seg1"


class TestChapterConstrainedBoost:
    """Tests for chapter-constrained candidate boost (US-71-011)."""

    @patch('src.matching.embedding_search.find_top_k_similar')
    def test_relevant_chapter_candidates_rank_higher(self, mock_find_similar):
        """Candidates from relevant video chapters rank higher than equal-similarity irrelevant ones."""
        # Two candidates with identical FAISS similarity scores
        mock_find_similar.return_value = (
            np.array([0.80, 0.80]),  # Same similarity
            np.array([0, 1])
        )

        seg0 = MockSRTSegment("relevant_ch")
        seg0.chapter_index = 0  # In relevant video chapter
        seg1 = MockSRTSegment("irrelevant_ch")
        seg1.chapter_index = 1  # In irrelevant video chapter

        config = EmbeddingSearchConfig(embedding_candidates=20)
        searcher = EmbeddingSearch(config, [[0.1, 0.2], [0.3, 0.4]], [seg0, seg1])

        # Relevance matrix: vo chapter 0 has high relevance to vid chapter 0, low to vid chapter 1
        relevance_matrix = [[0.8, 0.0]]
        results = searcher.search(
            [0.1, 0.2],
            relevance_matrix=relevance_matrix,
            voiceover_chapter_index=0,
            relevance_boost_weight=0.1,
        )

        assert len(results) == 2
        # seg0 (relevant chapter) should rank first due to boost
        assert results[0][0].id == "relevant_ch"
        assert results[1][0].id == "irrelevant_ch"
        # seg0 boosted: 0.80 + 0.8*0.1 = 0.88, seg1 unchanged: 0.80
        assert results[0][1] == pytest.approx(0.88)
        assert results[1][1] == pytest.approx(0.80)

    @patch('src.matching.embedding_search.find_top_k_similar')
    def test_boost_proportional_to_relevance_and_weight(self, mock_find_similar):
        """Boost = relevance_score * relevance_boost_weight."""
        mock_find_similar.return_value = (
            np.array([0.70]),
            np.array([0])
        )

        seg = MockSRTSegment("seg0")
        seg.chapter_index = 0

        config = EmbeddingSearchConfig(embedding_candidates=20)
        searcher = EmbeddingSearch(config, [[0.1]], [seg])

        relevance_matrix = [[0.5]]
        results = searcher.search(
            [0.1],
            relevance_matrix=relevance_matrix,
            voiceover_chapter_index=0,
            relevance_boost_weight=0.2,
        )

        # 0.70 + 0.5 * 0.2 = 0.80
        assert results[0][1] == pytest.approx(0.80)

    @patch('src.matching.embedding_search.find_top_k_similar')
    def test_no_chapter_info_neutral_treatment(self, mock_find_similar):
        """Candidates without chapter_index get no boost or penalty."""
        mock_find_similar.return_value = (
            np.array([0.80, 0.75]),
            np.array([0, 1])
        )

        seg0 = MockSRTSegment("no_chapter")
        # seg0 has no chapter_index attribute
        seg1 = MockSRTSegment("with_chapter")
        seg1.chapter_index = 0

        config = EmbeddingSearchConfig(embedding_candidates=20)
        searcher = EmbeddingSearch(config, [[0.1], [0.2]], [seg0, seg1])

        relevance_matrix = [[0.9]]
        results = searcher.search(
            [0.1],
            relevance_matrix=relevance_matrix,
            voiceover_chapter_index=0,
            relevance_boost_weight=0.1,
        )

        # seg0 (no chapter) stays at 0.80 — neutral
        # seg1 (chapter 0) gets boost: 0.75 + 0.9*0.1 = 0.84
        assert len(results) == 2
        # seg1 should now rank higher due to boost
        assert results[0][0].id == "with_chapter"
        assert results[0][1] == pytest.approx(0.84)
        assert results[1][0].id == "no_chapter"
        assert results[1][1] == pytest.approx(0.80)

    @patch('src.matching.embedding_search.find_top_k_similar')
    def test_no_relevance_matrix_no_boost(self, mock_find_similar):
        """When no relevance_matrix is provided, scores are unchanged."""
        mock_find_similar.return_value = (
            np.array([0.90]),
            np.array([0])
        )

        seg = MockSRTSegment("seg0")
        seg.chapter_index = 0

        config = EmbeddingSearchConfig(embedding_candidates=20)
        searcher = EmbeddingSearch(config, [[0.1]], [seg])

        results = searcher.search([0.1])  # No relevance params
        assert results[0][1] == pytest.approx(0.90)

    @patch('src.matching.embedding_search.find_top_k_similar')
    def test_negative_voiceover_chapter_no_boost(self, mock_find_similar):
        """When voiceover_chapter_index is -1, no boost applied."""
        mock_find_similar.return_value = (
            np.array([0.90]),
            np.array([0])
        )

        seg = MockSRTSegment("seg0")
        seg.chapter_index = 0

        config = EmbeddingSearchConfig(embedding_candidates=20)
        searcher = EmbeddingSearch(config, [[0.1]], [seg])

        results = searcher.search(
            [0.1],
            relevance_matrix=[[1.0]],
            voiceover_chapter_index=-1,
        )
        assert results[0][1] == pytest.approx(0.90)

    @patch('src.matching.embedding_search.find_top_k_similar')
    def test_post_retrieval_not_filter(self, mock_find_similar):
        """Boost is applied after FAISS retrieval, not as a filter — all candidates returned."""
        mock_find_similar.return_value = (
            np.array([0.90, 0.60, 0.30]),
            np.array([0, 1, 2])
        )

        segments = []
        for i in range(3):
            s = MockSRTSegment(f"seg{i}")
            s.chapter_index = i
            segments.append(s)

        config = EmbeddingSearchConfig(embedding_candidates=20)
        searcher = EmbeddingSearch(config, [[0.1], [0.2], [0.3]], segments)

        # Only vid chapter 2 is relevant
        relevance_matrix = [[0.0, 0.0, 1.0]]
        results = searcher.search(
            [0.1],
            relevance_matrix=relevance_matrix,
            voiceover_chapter_index=0,
            relevance_boost_weight=0.1,
        )

        # All 3 candidates returned (not filtered)
        assert len(results) == 3
        # seg2 boosted from 0.30 to 0.40, but seg0 at 0.90 still highest
        ids = [r[0].id for r in results]
        assert "seg0" in ids
        assert "seg1" in ids
        assert "seg2" in ids


class TestEmbeddingSearchSimilarity:
    """Tests for similarity calculation with known values."""

    @patch('src.matching.embedding_search.find_top_k_similar')
    def test_similarity_scores_preserved(self, mock_find_similar):
        """Test similarity scores from find_top_k_similar are preserved."""
        known_distances = np.array([0.99, 0.75, 0.50, 0.25])
        known_indices = np.array([0, 1, 2, 3])
        mock_find_similar.return_value = (known_distances, known_indices)

        config = EmbeddingSearchConfig()
        segments = [MockSRTSegment(f"seg{i}") for i in range(4)]
        searcher = EmbeddingSearch(config, [[0.1] * 4] * 4, segments)

        results = searcher.search([0.1, 0.1, 0.1, 0.1])

        # Verify scores match expected values
        assert results[0][1] == pytest.approx(0.99)
        assert results[1][1] == pytest.approx(0.75)
        assert results[2][1] == pytest.approx(0.50)
        assert results[3][1] == pytest.approx(0.25)

    @patch('src.matching.embedding_search.find_top_k_similar')
    def test_compute_similarity_passes_index_when_provided(self, mock_find_similar):
        """Test _compute_similarity passes FAISS index to find_top_k_similar."""
        mock_find_similar.return_value = (np.array([0.9]), np.array([0]))
        mock_index = Mock()

        config = EmbeddingSearchConfig()
        searcher = EmbeddingSearch(
            config, [[0.1]], [MockSRTSegment("seg0")], embedding_index=mock_index
        )

        searcher.search([0.1])

        # Verify index was passed to find_top_k_similar
        call_kwargs = mock_find_similar.call_args
        assert call_kwargs[1]['index'] is mock_index

    @patch('src.matching.embedding_search.find_top_k_similar')
    def test_compute_similarity_works_without_index(self, mock_find_similar):
        """Test _compute_similarity works when no FAISS index provided."""
        mock_find_similar.return_value = (np.array([0.85]), np.array([0]))

        config = EmbeddingSearchConfig()
        searcher = EmbeddingSearch(config, [[0.1]], [MockSRTSegment("seg0")])

        results = searcher.search([0.1])

        # Should work and return result
        assert len(results) == 1
        assert results[0][1] == pytest.approx(0.85)
        # Verify index=None was passed
        call_kwargs = mock_find_similar.call_args
        assert call_kwargs[1]['index'] is None


# Integration test without mocking (uses actual find_top_k_similar)
class TestEmbeddingSearchIntegration:
    """Integration tests using actual similarity computation."""

    def test_real_similarity_computation(self):
        """Test actual cosine similarity with known vectors."""
        config = EmbeddingSearchConfig(embedding_candidates=20)

        # Create embeddings with known similarity properties
        # Vector [1,0,0] should be most similar to [1,0,0]
        # Vector [0,1,0] should be orthogonal (similarity ~0)
        embeddings = [
            [1.0, 0.0, 0.0],  # seg0 - identical to query
            [0.0, 1.0, 0.0],  # seg1 - orthogonal
            [0.7, 0.7, 0.0],  # seg2 - partial similarity
        ]
        segments = [MockSRTSegment(f"seg{i}") for i in range(3)]

        searcher = EmbeddingSearch(config, embeddings, segments)
        query = [1.0, 0.0, 0.0]

        results = searcher.search(query)

        # seg0 should have highest similarity (cosine = 1.0)
        # seg2 should have medium similarity (cosine ~ 0.7)
        # seg1 should have lowest similarity (cosine = 0)
        assert len(results) == 3
        # First result should be seg0 with similarity close to 1
        assert results[0][0].id == "seg0"
        assert results[0][1] > 0.9
        # seg2 should be second with partial similarity
        seg_ids = [r[0].id for r in results]
        assert "seg0" in seg_ids
        assert "seg1" in seg_ids
        assert "seg2" in seg_ids
