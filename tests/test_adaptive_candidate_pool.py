"""
Unit tests for adaptive candidate pool sizing based on voiceover segment complexity (US-84-003).

Validates:
- Complexity score calculation from word count, entities, and location words
- Pool expansion for short generic segments
- Standard pool for complex specific segments
- Config fields min_candidates, max_candidates, complexity_scaling_factor
- from_matching_config reads new fields
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.matching.embedding_search import EmbeddingSearch, EmbeddingSearchConfig


class MockSRTSegment:
    """Mock segment for testing."""
    def __init__(self, segment_id: str, text: str = "test", source_file: str = "vid1"):
        self.id = segment_id
        self.text = text
        self.source_file = source_file


class TestComputeComplexityScore:
    """Test _compute_complexity_score returns correct scores for various inputs."""

    def _make_searcher(self):
        config = EmbeddingSearchConfig(embedding_candidates=20)
        return EmbeddingSearch(config, [], [])

    def test_empty_text_returns_zero(self):
        s = self._make_searcher()
        assert s._compute_complexity_score("") == 0.0
        assert s._compute_complexity_score("   ") == 0.0

    def test_short_generic_segment_low_complexity(self):
        """Short segments like 'the city' should have low complexity."""
        s = self._make_searcher()
        score = s._compute_complexity_score("the city")
        assert score < 0.3, f"Short generic segment should have low complexity, got {score}"

    def test_specific_segment_high_complexity(self):
        """Specific segments with entities and many words should have high complexity."""
        s = self._make_searcher()
        score = s._compute_complexity_score(
            "the historic architecture of Dubrovnik along the Adriatic Sea coastline"
        )
        assert score > 0.5, f"Specific segment should have high complexity, got {score}"

    def test_entity_presence_increases_complexity(self):
        """Capitalized words (not sentence-start) should increase complexity."""
        s = self._make_searcher()
        no_entities = s._compute_complexity_score("the old town walls and streets")
        with_entities = s._compute_complexity_score("the old Dubrovnik walls and Croatia streets")
        assert with_entities > no_entities

    def test_location_words_increase_complexity(self):
        """Location words (city, river, mountain) should increase complexity."""
        s = self._make_searcher()
        no_location = s._compute_complexity_score("beautiful sunny morning today here")
        with_location = s._compute_complexity_score("beautiful city mountain river today here")
        assert with_location > no_location

    def test_score_clamped_0_to_1(self):
        """Score should always be between 0.0 and 1.0."""
        s = self._make_searcher()
        # Very long specific text
        score = s._compute_complexity_score(
            "The ancient Roman Cathedral of Saint Domnius in Split Croatia "
            "overlooking the beautiful Adriatic Sea with mountain views from "
            "the palace tower near the harbor square"
        )
        assert 0.0 <= score <= 1.0


class TestAdaptivePoolSizing:
    """Test that search() adapts candidate pool size based on complexity."""

    def _make_searcher(self, embedding_candidates=20, min_k=15, max_k=100, scaling=0.5):
        config = EmbeddingSearchConfig(
            embedding_candidates=embedding_candidates,
            min_candidates=min_k,
            max_candidates=max_k,
            complexity_scaling_factor=scaling,
        )
        segments = [MockSRTSegment(f"seg{i}") for i in range(200)]
        embeddings = [[0.1] * 128 for _ in range(200)]
        return EmbeddingSearch(config, embeddings, segments)

    @patch.object(EmbeddingSearch, '_compute_similarity')
    def test_short_generic_gets_expanded_pool(self, mock_sim):
        """Short generic text ('the city') should get expanded pool (more candidates)."""
        searcher = self._make_searcher(embedding_candidates=20)
        # Return enough fake results
        mock_sim.return_value = (list(range(200)), list(range(200)))

        # Capture the k passed to _compute_similarity
        searcher.search(
            query_embedding=[0.1] * 128,
            voiceover_text="the city",
        )

        # _compute_similarity is called with fetch_k which is k*pre_fetch_multiplier (due to dedup)
        call_k = mock_sim.call_args[0][1]
        # "the city" is low complexity -> inverse_complexity high -> k should be > base 20
        effective_k = call_k // 2  # undo the *3 dedup multiplier
        assert effective_k > 20, f"Short generic should expand pool beyond 20, got {effective_k}"

    @patch.object(EmbeddingSearch, '_compute_similarity')
    def test_specific_segment_gets_standard_pool(self, mock_sim):
        """Specific segment should get smaller/standard pool."""
        searcher = self._make_searcher(embedding_candidates=20)
        mock_sim.return_value = (list(range(200)), list(range(200)))

        searcher.search(
            query_embedding=[0.1] * 128,
            voiceover_text="historic architecture of Dubrovnik along the Adriatic Sea coastline",
        )

        call_k = mock_sim.call_args[0][1]
        effective_k = call_k // 2

        # Also get short generic k for comparison
        searcher.search(
            query_embedding=[0.1] * 128,
            voiceover_text="the city",
        )
        generic_call_k = mock_sim.call_args[0][1]
        generic_effective_k = generic_call_k // 2

        assert effective_k < generic_effective_k, (
            f"Specific segment should get fewer candidates ({effective_k}) "
            f"than generic ({generic_effective_k})"
        )

    @patch.object(EmbeddingSearch, '_compute_similarity')
    def test_pool_respects_min_candidates(self, mock_sim):
        """Pool size should never go below min_candidates."""
        searcher = self._make_searcher(embedding_candidates=10, min_k=15)
        mock_sim.return_value = (list(range(200)), list(range(200)))

        # Even with very high complexity (small pool), should respect min
        searcher.search(
            query_embedding=[0.1] * 128,
            voiceover_text="historic architecture of Dubrovnik along the Adriatic Sea coastline",
        )

        call_k = mock_sim.call_args[0][1]
        # The k passed should be at least 20 (the max(k, 20) floor) * pre_fetch_multiplier (default 2)
        assert call_k >= 20 * 2

    @patch.object(EmbeddingSearch, '_compute_similarity')
    def test_pool_respects_max_candidates(self, mock_sim):
        """Pool size should never exceed max_candidates."""
        searcher = self._make_searcher(embedding_candidates=200, max_k=50, scaling=0.5)
        mock_sim.return_value = (list(range(200)), list(range(200)))

        searcher.search(
            query_embedding=[0.1] * 128,
            voiceover_text="the city",
        )

        call_k = mock_sim.call_args[0][1]
        effective_k = call_k // 2
        assert effective_k <= 50, f"Should not exceed max_candidates=50, got {effective_k}"

    @patch.object(EmbeddingSearch, '_compute_similarity')
    def test_num_candidates_override_bypasses_adaptive(self, mock_sim):
        """Explicit num_candidates should override adaptive sizing."""
        searcher = self._make_searcher(embedding_candidates=20)
        mock_sim.return_value = (list(range(200)), list(range(200)))

        searcher.search(
            query_embedding=[0.1] * 128,
            num_candidates=42,
            voiceover_text="the city",  # Should be ignored when num_candidates set
        )

        call_k = mock_sim.call_args[0][1]
        effective_k = call_k // 2
        assert effective_k == 42, f"num_candidates override should be 42, got {effective_k}"


class TestConfigFields:
    """Test new config fields exist and are read correctly."""

    def test_default_config_values(self):
        config = EmbeddingSearchConfig()
        assert config.min_candidates == 15
        assert config.max_candidates == 100
        assert config.complexity_scaling_factor == 0.5

    def test_from_matching_config_reads_new_fields(self):
        mock_config = Mock()
        mock_config.embedding_candidates = 30
        mock_config.max_candidates_per_source = 3
        mock_config.min_candidates = 10
        mock_config.max_candidates = 80
        mock_config.complexity_scaling_factor = 0.7

        searcher = EmbeddingSearch.from_matching_config(mock_config, [], [])
        assert searcher.config.min_candidates == 10
        assert searcher.config.max_candidates == 80
        assert searcher.config.complexity_scaling_factor == 0.7

    def test_from_matching_config_defaults_when_missing(self):
        """When matching config doesn't have new fields, use defaults."""
        mock_config = Mock(spec=[])  # No attributes
        mock_config.embedding_candidates = 20
        # getattr with defaults should work even without the fields
        searcher = EmbeddingSearch.from_matching_config(mock_config, [], [])
        assert searcher.config.min_candidates == 15
        assert searcher.config.max_candidates == 100
        assert searcher.config.complexity_scaling_factor == 0.5
