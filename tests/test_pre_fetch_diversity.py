"""
Tests for pre-fetch diversity filter in FAISS candidate retrieval (US-84-005).

Validates that EmbeddingSearch fetches k*pre_fetch_multiplier candidates from FAISS,
then applies source diversity filtering to return k diverse results.
"""

import logging
import pytest
import numpy as np
from unittest.mock import patch

from src.matching.embedding_search import EmbeddingSearch, EmbeddingSearchConfig


class MockSegment:
    """Mock segment with source_file for diversity testing."""
    def __init__(self, segment_id: str, source_file: str, text: str = "test"):
        self.id = segment_id
        self.source_file = source_file
        self.text = text
        self.video_id = source_file


class TestPreFetchMultiplier:
    """Tests for the pre_fetch_multiplier config and FAISS fetch behavior."""

    def test_default_pre_fetch_multiplier_is_2(self):
        """Config default pre_fetch_multiplier should be 2."""
        config = EmbeddingSearchConfig()
        assert config.pre_fetch_multiplier == 2

    @patch('src.matching.embedding_search.find_top_k_similar')
    def test_fetch_k_uses_pre_fetch_multiplier(self, mock_find):
        """When max_candidates_per_source > 0, fetch_k = k * pre_fetch_multiplier."""
        mock_find.return_value = (np.array([0.9]), np.array([0]))

        config = EmbeddingSearchConfig(
            embedding_candidates=20,
            max_candidates_per_source=3,
            pre_fetch_multiplier=2,
        )
        segments = [MockSegment("seg0", "vidA")]
        searcher = EmbeddingSearch(config, [[0.1]], segments)
        searcher.search([0.1])

        # k=20 (min 20), multiplier=2, so fetch_k=40
        called_k = mock_find.call_args[0][2]
        assert called_k == 40

    @patch('src.matching.embedding_search.find_top_k_similar')
    def test_fetch_k_no_multiplier_when_dedup_disabled(self, mock_find):
        """When max_candidates_per_source = 0, fetch_k = k (no multiplier)."""
        mock_find.return_value = (np.array([0.9]), np.array([0]))

        config = EmbeddingSearchConfig(
            embedding_candidates=20,
            max_candidates_per_source=0,
            pre_fetch_multiplier=2,
        )
        segments = [MockSegment("seg0", "vidA")]
        searcher = EmbeddingSearch(config, [[0.1]], segments)
        searcher.search([0.1])

        called_k = mock_find.call_args[0][2]
        assert called_k == 20

    @patch('src.matching.embedding_search.find_top_k_similar')
    def test_custom_pre_fetch_multiplier_3(self, mock_find):
        """Custom pre_fetch_multiplier=3 should fetch k*3."""
        mock_find.return_value = (np.array([0.9]), np.array([0]))

        config = EmbeddingSearchConfig(
            embedding_candidates=20,
            max_candidates_per_source=3,
            pre_fetch_multiplier=3,
        )
        segments = [MockSegment("seg0", "vidA")]
        searcher = EmbeddingSearch(config, [[0.1]], segments)
        searcher.search([0.1])

        called_k = mock_find.call_args[0][2]
        assert called_k == 60  # 20 * 3


class TestDiverseResultsFromDominatedFAISS:
    """Test that diverse results are returned when top FAISS results are dominated by few sources."""

    @patch('src.matching.embedding_search.find_top_k_similar')
    def test_dominated_sources_still_yield_diverse_results(self, mock_find):
        """When top-20 FAISS results are dominated by 2 sources (10 each),
        pre-fetch multiplier retrieves extra candidates to find diverse ones."""
        # Simulate 40 candidates: 10 from vidA (scores 0.95-0.86), 10 from vidB (0.85-0.76),
        # then 20 from diverse sources vidC-vidV (scores 0.75-0.56)
        num_candidates = 40
        scores = []
        indices = []
        segments = []

        # 10 from vidA (highest scores)
        for i in range(10):
            scores.append(0.95 - i * 0.01)
            indices.append(i)
            segments.append(MockSegment(f"segA{i}", "vidA"))

        # 10 from vidB
        for i in range(10):
            scores.append(0.85 - i * 0.01)
            indices.append(10 + i)
            segments.append(MockSegment(f"segB{i}", "vidB"))

        # 20 from diverse sources (vidC through vidV)
        for i in range(20):
            scores.append(0.75 - i * 0.01)
            indices.append(20 + i)
            segments.append(MockSegment(f"segD{i}", f"vid{chr(67 + i)}"))

        mock_find.return_value = (
            np.array(scores),
            np.array(indices),
        )

        config = EmbeddingSearchConfig(
            embedding_candidates=20,
            max_candidates_per_source=3,
            pre_fetch_multiplier=2,
        )
        embeddings = [[0.1] * 3] * num_candidates
        searcher = EmbeddingSearch(config, embeddings, segments)
        results = searcher.search([0.1] * 3)

        # Should have at most 20 results (target k=20)
        assert len(results) <= 20

        # Count unique sources in results
        result_sources = set()
        for seg, _ in results:
            result_sources.add(seg.source_file)

        # vidA and vidB should each have at most 3 candidates
        source_counts = {}
        for seg, _ in results:
            src = seg.source_file
            source_counts[src] = source_counts.get(src, 0) + 1

        assert source_counts.get("vidA", 0) <= 3
        assert source_counts.get("vidB", 0) <= 3

        # Should have diverse sources beyond just vidA and vidB
        assert len(result_sources) > 2, (
            f"Expected diverse sources but only got {result_sources}"
        )

        # With max_per_source=3, we need at least 20/3 ~= 7 unique sources to fill 20 slots
        # The diversity filter should pull in sources from the extended fetch
        assert len(result_sources) >= 5, (
            f"Expected at least 5 unique sources, got {len(result_sources)}: {result_sources}"
        )

    @patch('src.matching.embedding_search.find_top_k_similar')
    def test_without_multiplier_misses_diverse_candidates(self, mock_find):
        """With pre_fetch_multiplier=1, dominated top-20 yields fewer diverse results."""
        # Only 20 candidates: 10 from vidA, 10 from vidB
        num_candidates = 20
        scores = []
        indices = []
        segments = []

        for i in range(10):
            scores.append(0.95 - i * 0.01)
            indices.append(i)
            segments.append(MockSegment(f"segA{i}", "vidA"))

        for i in range(10):
            scores.append(0.85 - i * 0.01)
            indices.append(10 + i)
            segments.append(MockSegment(f"segB{i}", "vidB"))

        mock_find.return_value = (
            np.array(scores),
            np.array(indices),
        )

        config = EmbeddingSearchConfig(
            embedding_candidates=20,
            max_candidates_per_source=3,
            pre_fetch_multiplier=1,  # No extra fetch
        )
        embeddings = [[0.1] * 3] * num_candidates
        searcher = EmbeddingSearch(config, embeddings, segments)
        results = searcher.search([0.1] * 3)

        # With only 2 sources and max 3 each, we get at most 6 results
        assert len(results) <= 6

        source_counts = {}
        for seg, _ in results:
            src = seg.source_file
            source_counts[src] = source_counts.get(src, 0) + 1

        # Only 2 sources available
        assert len(source_counts) == 2


class TestSourceConcentrationLogging:
    """Test source concentration ratio logging (US-84-005)."""

    @patch('src.matching.embedding_search.find_top_k_similar')
    def test_low_concentration_logs_warning(self, mock_find, caplog):
        """When concentration ratio < 0.3, a warning should be logged."""
        # 10 candidates from only 2 sources = 0.2 ratio
        scores = [0.9 - i * 0.01 for i in range(10)]
        indices = list(range(10))
        segments = []
        for i in range(5):
            segments.append(MockSegment(f"segA{i}", "vidA"))
        for i in range(5):
            segments.append(MockSegment(f"segB{i}", "vidB"))

        mock_find.return_value = (np.array(scores), np.array(indices))

        config = EmbeddingSearchConfig(
            embedding_candidates=10,
            max_candidates_per_source=0,  # Disable dedup to test logging only
        )
        embeddings = [[0.1] * 3] * 10
        searcher = EmbeddingSearch(config, embeddings, segments)

        with caplog.at_level(logging.WARNING, logger='src.matching.embedding_search'):
            searcher.search([0.1] * 3)

        assert any("Low source diversity" in msg for msg in caplog.messages), (
            f"Expected 'Low source diversity' warning, got: {caplog.messages}"
        )

    @patch('src.matching.embedding_search.find_top_k_similar')
    def test_high_concentration_no_warning(self, mock_find, caplog):
        """When concentration ratio >= 0.3, no warning should be logged."""
        # 10 candidates from 5 sources = 0.5 ratio
        scores = [0.9 - i * 0.01 for i in range(10)]
        indices = list(range(10))
        segments = []
        for i in range(10):
            segments.append(MockSegment(f"seg{i}", f"vid{i % 5}"))

        mock_find.return_value = (np.array(scores), np.array(indices))

        config = EmbeddingSearchConfig(
            embedding_candidates=10,
            max_candidates_per_source=0,
        )
        embeddings = [[0.1] * 3] * 10
        searcher = EmbeddingSearch(config, embeddings, segments)

        with caplog.at_level(logging.WARNING, logger='src.matching.embedding_search'):
            searcher.search([0.1] * 3)

        assert not any("Low source diversity" in msg for msg in caplog.messages), (
            f"Unexpected warning logged: {caplog.messages}"
        )

    @patch('src.matching.embedding_search.find_top_k_similar')
    def test_concentration_ratio_logged_at_debug(self, mock_find, caplog):
        """Concentration ratio should be logged at DEBUG level."""
        scores = [0.9 - i * 0.01 for i in range(6)]
        indices = list(range(6))
        segments = [MockSegment(f"seg{i}", f"vid{i}") for i in range(6)]

        mock_find.return_value = (np.array(scores), np.array(indices))

        config = EmbeddingSearchConfig(
            embedding_candidates=6,
            max_candidates_per_source=0,
        )
        embeddings = [[0.1] * 3] * 6
        searcher = EmbeddingSearch(config, embeddings, segments)

        with caplog.at_level(logging.DEBUG, logger='src.matching.embedding_search'):
            searcher.search([0.1] * 3)

        assert any("Source concentration" in msg for msg in caplog.messages), (
            f"Expected 'Source concentration' debug log, got: {caplog.messages}"
        )


class TestFromMatchingConfigPreFetchMultiplier:
    """Test that from_matching_config reads pre_fetch_multiplier."""

    def test_reads_pre_fetch_multiplier_from_config(self):
        """from_matching_config should read pre_fetch_multiplier."""
        from unittest.mock import Mock
        mock_config = Mock()
        mock_config.embedding_candidates = 20
        mock_config.max_candidates_per_source = 3
        mock_config.min_candidates = 15
        mock_config.max_candidates = 100
        mock_config.complexity_scaling_factor = 0.5
        mock_config.pre_fetch_multiplier = 4

        searcher = EmbeddingSearch.from_matching_config(
            mock_config, [[0.1]], [MockSegment("s", "v")]
        )
        assert searcher.config.pre_fetch_multiplier == 4

    def test_defaults_pre_fetch_multiplier_to_2(self):
        """When pre_fetch_multiplier is missing from config, default to 2."""
        from unittest.mock import Mock
        mock_config = Mock(spec=[])
        mock_config.embedding_candidates = 20
        mock_config.max_candidates_per_source = 3

        searcher = EmbeddingSearch.from_matching_config(
            mock_config, [[0.1]], [MockSegment("s", "v")]
        )
        assert searcher.config.pre_fetch_multiplier == 2
