"""
Tests for semantic coherence (topic flow between adjacent segments).

Tests the compute_semantic_coherence() function in src/matching/scoring.py,
which evaluates embedding similarity between current and previous matches
to score topic flow and apply confidence adjustments.

US-011: Add semantic coherence between adjacent segments
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from src.matching.scoring import (
    compute_semantic_coherence,
    SEMANTIC_COHERENCE_SMOOTH_THRESHOLD,
    SEMANTIC_COHERENCE_ABRUPT_THRESHOLD,
    SEMANTIC_COHERENCE_SMOOTH_BOOST,
    SEMANTIC_COHERENCE_ABRUPT_PENALTY,
)


class TestConstants:
    """Tests for semantic coherence constants."""

    @pytest.mark.fast
    def test_smooth_threshold_value(self):
        """Smooth flow threshold is 0.6."""
        assert SEMANTIC_COHERENCE_SMOOTH_THRESHOLD == 0.6

    @pytest.mark.fast
    def test_abrupt_threshold_value(self):
        """Abrupt flow threshold is 0.3."""
        assert SEMANTIC_COHERENCE_ABRUPT_THRESHOLD == 0.3

    @pytest.mark.fast
    def test_smooth_boost_value(self):
        """Smooth flow boost is +0.03."""
        assert SEMANTIC_COHERENCE_SMOOTH_BOOST == 0.03

    @pytest.mark.fast
    def test_abrupt_penalty_value(self):
        """Abrupt flow penalty is 0.05."""
        assert SEMANTIC_COHERENCE_ABRUPT_PENALTY == 0.05

    @pytest.mark.fast
    def test_thresholds_ordering(self):
        """Abrupt threshold should be less than smooth threshold."""
        assert SEMANTIC_COHERENCE_ABRUPT_THRESHOLD < SEMANTIC_COHERENCE_SMOOTH_THRESHOLD


class TestDisabledMode:
    """Tests for when semantic coherence is disabled."""

    @pytest.mark.fast
    def test_disabled_returns_zero_adjustment(self):
        """When disabled, returns 0.0 adjustment."""
        current_emb = np.array([1.0, 0.0, 0.0])
        previous_emb = np.array([0.0, 1.0, 0.0])

        adjustment, reason = compute_semantic_coherence(
            current_emb, previous_emb, semantic_coherence_enabled=False
        )

        assert adjustment == 0.0
        assert reason == "semantic_coherence_disabled"

    @pytest.mark.fast
    def test_disabled_with_similar_embeddings(self):
        """When disabled, similar embeddings still get no adjustment."""
        emb = np.array([1.0, 0.0, 0.0])

        adjustment, reason = compute_semantic_coherence(
            emb, emb, semantic_coherence_enabled=False
        )

        assert adjustment == 0.0
        assert reason == "semantic_coherence_disabled"


class TestMissingEmbeddings:
    """Tests for handling None/missing embeddings."""

    @pytest.mark.fast
    def test_none_current_embedding(self):
        """None current embedding returns 0.0 with reason."""
        previous_emb = np.array([1.0, 0.0, 0.0])

        adjustment, reason = compute_semantic_coherence(
            None, previous_emb, semantic_coherence_enabled=True
        )

        assert adjustment == 0.0
        assert reason == "missing_embedding"

    @pytest.mark.fast
    def test_none_previous_embedding(self):
        """None previous embedding returns 0.0 with reason."""
        current_emb = np.array([1.0, 0.0, 0.0])

        adjustment, reason = compute_semantic_coherence(
            current_emb, None, semantic_coherence_enabled=True
        )

        assert adjustment == 0.0
        assert reason == "missing_embedding"

    @pytest.mark.fast
    def test_both_embeddings_none(self):
        """Both embeddings None returns 0.0 with reason."""
        adjustment, reason = compute_semantic_coherence(
            None, None, semantic_coherence_enabled=True
        )

        assert adjustment == 0.0
        assert reason == "missing_embedding"


class TestSmoothTopicFlow:
    """Tests for smooth topic flow detection (similarity > 0.6)."""

    @pytest.mark.fast
    def test_identical_embeddings_get_boost(self):
        """Identical embeddings (similarity=1.0) get smooth flow boost."""
        emb = np.array([1.0, 0.0, 0.0])

        adjustment, reason = compute_semantic_coherence(
            emb, emb, semantic_coherence_enabled=True
        )

        assert adjustment == SEMANTIC_COHERENCE_SMOOTH_BOOST
        assert "smooth_topic_flow" in reason
        assert "+0.03" in reason

    @pytest.mark.fast
    def test_high_similarity_embeddings_get_boost(self):
        """High similarity embeddings (sim=0.9) get smooth flow boost."""
        # Create embeddings with ~0.9 cosine similarity
        current_emb = np.array([1.0, 0.0, 0.0])
        previous_emb = np.array([0.95, 0.31, 0.0])  # ~0.95 similarity

        adjustment, reason = compute_semantic_coherence(
            current_emb, previous_emb, semantic_coherence_enabled=True
        )

        assert adjustment == SEMANTIC_COHERENCE_SMOOTH_BOOST
        assert "smooth_topic_flow" in reason

    @pytest.mark.fast
    def test_similarity_just_above_threshold(self):
        """Similarity just above 0.6 threshold gets boost."""
        # Create embeddings with similarity ~0.65
        current_emb = np.array([1.0, 0.0, 0.0])
        previous_emb = np.array([0.65, 0.76, 0.0])  # ~0.65 similarity when normalized

        # Actually calculate the similarity
        from src.embeddings import cosine_similarity
        sim = cosine_similarity(current_emb, previous_emb)

        adjustment, reason = compute_semantic_coherence(
            current_emb, previous_emb, semantic_coherence_enabled=True
        )

        if sim > SEMANTIC_COHERENCE_SMOOTH_THRESHOLD:
            assert adjustment == SEMANTIC_COHERENCE_SMOOTH_BOOST
            assert "smooth_topic_flow" in reason


class TestAbruptTopicFlow:
    """Tests for abrupt topic flow detection (similarity < 0.3)."""

    @pytest.mark.fast
    def test_orthogonal_embeddings_get_penalty(self):
        """Orthogonal embeddings (similarity=0) get abrupt flow penalty."""
        current_emb = np.array([1.0, 0.0, 0.0])
        previous_emb = np.array([0.0, 1.0, 0.0])

        adjustment, reason = compute_semantic_coherence(
            current_emb, previous_emb, semantic_coherence_enabled=True
        )

        assert adjustment == -SEMANTIC_COHERENCE_ABRUPT_PENALTY
        assert "abrupt_topic_flow" in reason
        assert "-0.05" in reason

    @pytest.mark.fast
    def test_opposite_embeddings_get_penalty(self):
        """Opposite embeddings (similarity=-1) get abrupt flow penalty."""
        current_emb = np.array([1.0, 0.0, 0.0])
        previous_emb = np.array([-1.0, 0.0, 0.0])

        adjustment, reason = compute_semantic_coherence(
            current_emb, previous_emb, semantic_coherence_enabled=True
        )

        assert adjustment == -SEMANTIC_COHERENCE_ABRUPT_PENALTY
        assert "abrupt_topic_flow" in reason

    @pytest.mark.fast
    def test_low_similarity_embeddings_get_penalty(self):
        """Low similarity embeddings (sim=0.2) get abrupt flow penalty."""
        # Create embeddings with ~0.2 cosine similarity
        current_emb = np.array([1.0, 0.0, 0.0])
        previous_emb = np.array([0.2, 0.98, 0.0])  # ~0.2 similarity

        adjustment, reason = compute_semantic_coherence(
            current_emb, previous_emb, semantic_coherence_enabled=True
        )

        assert adjustment == -SEMANTIC_COHERENCE_ABRUPT_PENALTY
        assert "abrupt_topic_flow" in reason


class TestNeutralTopicFlow:
    """Tests for neutral topic flow (0.3 <= similarity <= 0.6)."""

    @pytest.mark.fast
    def test_medium_similarity_no_adjustment(self):
        """Medium similarity embeddings (sim=0.5) get no adjustment."""
        # Create embeddings with ~0.5 cosine similarity
        current_emb = np.array([1.0, 0.0, 0.0])
        previous_emb = np.array([0.5, 0.866, 0.0])  # ~0.5 similarity

        adjustment, reason = compute_semantic_coherence(
            current_emb, previous_emb, semantic_coherence_enabled=True
        )

        assert adjustment == 0.0
        assert "neutral_topic_flow" in reason

    @pytest.mark.fast
    def test_similarity_at_lower_boundary(self):
        """Similarity at exactly 0.3 is neutral (not abrupt)."""
        # Create embeddings with ~0.3 cosine similarity
        current_emb = np.array([1.0, 0.0, 0.0])
        previous_emb = np.array([0.3, 0.954, 0.0])  # ~0.3 similarity

        from src.embeddings import cosine_similarity
        sim = cosine_similarity(current_emb, previous_emb)

        adjustment, reason = compute_semantic_coherence(
            current_emb, previous_emb, semantic_coherence_enabled=True
        )

        # At exactly 0.3, it's not abrupt (< 0.3 is abrupt, not <=)
        if abs(sim - 0.3) < 0.05:  # Allow some tolerance
            # Near boundary - check for correct behavior
            if sim < SEMANTIC_COHERENCE_ABRUPT_THRESHOLD:
                assert adjustment == -SEMANTIC_COHERENCE_ABRUPT_PENALTY
            else:
                assert adjustment == 0.0

    @pytest.mark.fast
    def test_similarity_at_upper_boundary(self):
        """Similarity at exactly 0.6 is neutral (not smooth)."""
        # Create embeddings with ~0.6 cosine similarity
        current_emb = np.array([1.0, 0.0, 0.0])
        previous_emb = np.array([0.6, 0.8, 0.0])  # ~0.6 similarity

        from src.embeddings import cosine_similarity
        sim = cosine_similarity(current_emb, previous_emb)

        adjustment, reason = compute_semantic_coherence(
            current_emb, previous_emb, semantic_coherence_enabled=True
        )

        # At exactly 0.6, it's not smooth (> 0.6 is smooth, not >=)
        if abs(sim - 0.6) < 0.05:  # Allow some tolerance
            if sim > SEMANTIC_COHERENCE_SMOOTH_THRESHOLD:
                assert adjustment == SEMANTIC_COHERENCE_SMOOTH_BOOST
            else:
                assert adjustment == 0.0


class TestListInputs:
    """Tests for list inputs (not just numpy arrays)."""

    @pytest.mark.fast
    def test_list_inputs_work(self):
        """Function works with Python lists, not just numpy arrays."""
        current_emb = [1.0, 0.0, 0.0]
        previous_emb = [1.0, 0.0, 0.0]  # Identical

        adjustment, reason = compute_semantic_coherence(
            current_emb, previous_emb, semantic_coherence_enabled=True
        )

        assert adjustment == SEMANTIC_COHERENCE_SMOOTH_BOOST
        assert "smooth_topic_flow" in reason

    @pytest.mark.fast
    def test_mixed_list_and_numpy(self):
        """Function works with mixed list and numpy array inputs."""
        current_emb = [1.0, 0.0, 0.0]
        previous_emb = np.array([1.0, 0.0, 0.0])

        adjustment, reason = compute_semantic_coherence(
            current_emb, previous_emb, semantic_coherence_enabled=True
        )

        assert adjustment == SEMANTIC_COHERENCE_SMOOTH_BOOST


class TestHighDimensionalEmbeddings:
    """Tests with high-dimensional embeddings (typical for models)."""

    @pytest.mark.fast
    def test_768_dim_embeddings(self):
        """Works with 768-dimensional embeddings (BERT-like)."""
        np.random.seed(42)
        current_emb = np.random.randn(768)
        # Create similar embedding by adding small noise
        previous_emb = current_emb + np.random.randn(768) * 0.1

        adjustment, reason = compute_semantic_coherence(
            current_emb, previous_emb, semantic_coherence_enabled=True
        )

        # High similarity due to small noise
        assert adjustment == SEMANTIC_COHERENCE_SMOOTH_BOOST
        assert "smooth_topic_flow" in reason

    @pytest.mark.fast
    def test_384_dim_embeddings(self):
        """Works with 384-dimensional embeddings (sentence-transformers)."""
        np.random.seed(123)
        current_emb = np.random.randn(384)
        # Create very different embedding
        previous_emb = np.random.randn(384)

        adjustment, reason = compute_semantic_coherence(
            current_emb, previous_emb, semantic_coherence_enabled=True
        )

        # Random embeddings have low similarity (near 0)
        # Should be abrupt or neutral
        assert adjustment in [0.0, -SEMANTIC_COHERENCE_ABRUPT_PENALTY]


class TestReasonStrings:
    """Tests for reason string formatting."""

    @pytest.mark.fast
    def test_smooth_reason_includes_similarity(self):
        """Smooth flow reason includes similarity value."""
        emb = np.array([1.0, 0.0, 0.0])

        adjustment, reason = compute_semantic_coherence(
            emb, emb, semantic_coherence_enabled=True
        )

        assert "sim=" in reason
        assert "1.000" in reason  # cos(0) = 1.0

    @pytest.mark.fast
    def test_abrupt_reason_includes_similarity(self):
        """Abrupt flow reason includes similarity value."""
        current_emb = np.array([1.0, 0.0, 0.0])
        previous_emb = np.array([0.0, 1.0, 0.0])

        adjustment, reason = compute_semantic_coherence(
            current_emb, previous_emb, semantic_coherence_enabled=True
        )

        assert "sim=" in reason
        assert "0.000" in reason  # orthogonal = 0

    @pytest.mark.fast
    def test_neutral_reason_includes_similarity(self):
        """Neutral flow reason includes similarity value."""
        current_emb = np.array([1.0, 0.0, 0.0])
        previous_emb = np.array([0.5, 0.866, 0.0])

        adjustment, reason = compute_semantic_coherence(
            current_emb, previous_emb, semantic_coherence_enabled=True
        )

        assert "sim=" in reason


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.fast
    def test_empty_array_handling(self):
        """Empty arrays are handled gracefully."""
        current_emb = np.array([])
        previous_emb = np.array([1.0, 0.0, 0.0])

        # Should not crash - empty array gives 0.0 similarity (cosine_similarity returns 0.0)
        adjustment, reason = compute_semantic_coherence(
            current_emb, previous_emb, semantic_coherence_enabled=True
        )

        # Empty array results in 0.0 similarity which is abrupt flow
        # This is acceptable behavior - the function handles it gracefully
        assert isinstance(adjustment, float)
        assert isinstance(reason, str)

    @pytest.mark.fast
    def test_mismatched_dimensions(self):
        """Mismatched embedding dimensions are handled."""
        current_emb = np.array([1.0, 0.0, 0.0])
        previous_emb = np.array([1.0, 0.0])  # Different dimension

        adjustment, reason = compute_semantic_coherence(
            current_emb, previous_emb, semantic_coherence_enabled=True
        )

        # Should handle gracefully
        assert isinstance(adjustment, float)
        assert isinstance(reason, str)

    @pytest.mark.fast
    def test_zero_vector_handling(self):
        """Zero vectors are handled gracefully."""
        current_emb = np.array([0.0, 0.0, 0.0])
        previous_emb = np.array([1.0, 0.0, 0.0])

        adjustment, reason = compute_semantic_coherence(
            current_emb, previous_emb, semantic_coherence_enabled=True
        )

        # Zero norm vector - cosine_similarity returns 0.0
        assert isinstance(adjustment, float)


class TestIntegrationWithConfig:
    """Tests for integration with MatchingConfig."""

    @pytest.mark.fast
    def test_config_option_exists(self):
        """semantic_coherence_enabled config option exists."""
        from src.config.sections.matching import MatchingConfig

        config = MatchingConfig()
        assert hasattr(config, 'semantic_coherence_enabled')
        assert config.semantic_coherence_enabled is True  # Default enabled

    @pytest.mark.fast
    def test_config_default_is_true(self):
        """semantic_coherence_enabled defaults to True."""
        from src.config.sections.matching import MatchingConfig

        config = MatchingConfig()
        assert config.semantic_coherence_enabled is True

    @pytest.mark.fast
    def test_config_can_be_disabled(self):
        """semantic_coherence_enabled can be set to False."""
        from src.config.sections.matching import MatchingConfig

        config = MatchingConfig(semantic_coherence_enabled=False)
        assert config.semantic_coherence_enabled is False


class TestSimilarityCalculation:
    """Tests verifying correct similarity calculation."""

    @pytest.mark.fast
    def test_cosine_similarity_range(self):
        """Cosine similarity should be between -1 and 1."""
        from src.embeddings import cosine_similarity

        # Test various embedding pairs
        test_cases = [
            (np.array([1.0, 0.0]), np.array([1.0, 0.0])),  # Same: 1.0
            (np.array([1.0, 0.0]), np.array([0.0, 1.0])),  # Orthogonal: 0.0
            (np.array([1.0, 0.0]), np.array([-1.0, 0.0])),  # Opposite: -1.0
        ]

        for emb1, emb2 in test_cases:
            sim = cosine_similarity(emb1, emb2)
            assert -1.0 <= sim <= 1.0

    @pytest.mark.fast
    def test_adjustment_applied_correctly(self):
        """Verify adjustment is applied based on actual similarity."""
        from src.embeddings import cosine_similarity

        # Test smooth flow case
        current = np.array([1.0, 0.0, 0.0])
        previous = np.array([0.95, 0.31, 0.0])

        sim = cosine_similarity(current, previous)
        adjustment, _ = compute_semantic_coherence(current, previous)

        if sim > 0.6:
            assert adjustment == 0.03
        elif sim < 0.3:
            assert adjustment == -0.05
        else:
            assert adjustment == 0.0
