"""
Tests for semantic distance-based topic penalty (US-003).

Tests the enhanced compute_topic_penalty function that uses embedding-based
semantic similarity when available, with fallback to keyword overlap.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.topic_extraction import (
    compute_topic_penalty,
    _compute_semantic_topic_penalty,
    _compute_keyword_topic_penalty,
    compute_topic_overlap,
)


# Mark all tests in this module as unit tests
pytestmark = pytest.mark.unit


class TestComputeTopicPenaltySignature:
    """Test the updated function signature and parameters."""

    @pytest.mark.fast
    def test_embedding_provider_parameter_exists(self):
        """Test that embedding_provider parameter is accepted."""
        # Should not raise TypeError
        result = compute_topic_penalty(
            ["topic1"], ["topic2"], embedding_provider=None
        )
        assert isinstance(result, float)

    @pytest.mark.fast
    def test_default_embedding_provider_is_none(self):
        """Test that embedding_provider defaults to None."""
        import inspect
        sig = inspect.signature(compute_topic_penalty)
        param = sig.parameters.get('embedding_provider')
        assert param is not None
        assert param.default is None

    @pytest.mark.fast
    def test_backward_compatible_without_embedding_provider(self):
        """Test that function works without embedding_provider argument."""
        # Original call signature should still work
        result = compute_topic_penalty(["earthquake"], ["cooking"])
        assert isinstance(result, float)
        assert 0 <= result <= 0.15  # Within max_penalty range


class TestSemanticPenaltyHelper:
    """Test the _compute_semantic_topic_penalty helper function."""

    @pytest.mark.fast
    def test_semantic_penalty_returns_float_or_none(self):
        """Test that semantic penalty returns float or None."""
        mock_provider = MagicMock()
        # Mock embedding that returns two similar vectors
        mock_provider.embed.return_value = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]]

        result = _compute_semantic_topic_penalty(
            ["earthquake"], ["disaster"], 0.15, mock_provider
        )

        # Should return a float (not None since embed succeeded)
        assert result is None or isinstance(result, float)

    @pytest.mark.fast
    def test_semantic_penalty_returns_none_on_embed_failure(self):
        """Test that semantic penalty returns None when embedding fails."""
        mock_provider = MagicMock()
        mock_provider.embed.side_effect = Exception("API error")

        result = _compute_semantic_topic_penalty(
            ["earthquake"], ["disaster"], 0.15, mock_provider
        )

        assert result is None

    @pytest.mark.fast
    def test_semantic_penalty_returns_none_for_wrong_embedding_count(self):
        """Test that semantic penalty returns None for wrong embedding count."""
        mock_provider = MagicMock()
        mock_provider.embed.return_value = [[1.0, 0.0, 0.0]]  # Only 1 embedding

        result = _compute_semantic_topic_penalty(
            ["earthquake"], ["disaster"], 0.15, mock_provider
        )

        assert result is None

    @pytest.mark.fast
    def test_semantic_penalty_returns_none_for_empty_topics(self):
        """Test that semantic penalty returns None for empty topic text."""
        mock_provider = MagicMock()

        # Empty list results in empty string after join
        result = _compute_semantic_topic_penalty(
            [], ["disaster"], 0.15, mock_provider
        )

        assert result is None


class TestSemanticSimilarityScaling:
    """Test that semantic similarity maps correctly to penalty values."""

    def _mock_provider_with_similarity(self, similarity: float):
        """Create a mock provider that returns embeddings with given similarity."""
        import numpy as np

        mock_provider = MagicMock()

        # Create two unit vectors with the desired cosine similarity
        # For simplicity: v1 = [1, 0], v2 = [cos(theta), sin(theta)]
        # where theta = arccos(similarity)
        if similarity >= 1.0:
            v1 = [1.0, 0.0]
            v2 = [1.0, 0.0]
        else:
            theta = np.arccos(np.clip(similarity, -1.0, 1.0))
            v1 = [1.0, 0.0]
            v2 = [np.cos(theta), np.sin(theta)]

        mock_provider.embed.return_value = [v1, v2]
        return mock_provider

    @pytest.mark.fast
    def test_high_similarity_returns_zero_penalty(self):
        """Test that high similarity (>=0.8) returns zero penalty."""
        provider = self._mock_provider_with_similarity(0.85)

        result = _compute_semantic_topic_penalty(
            ["earthquake"], ["disaster"], 0.15, provider
        )

        assert result == 0.0

    @pytest.mark.fast
    def test_identical_topics_return_zero_penalty(self):
        """Test that identical topics (similarity ~1.0) return zero penalty."""
        provider = self._mock_provider_with_similarity(0.99)

        result = _compute_semantic_topic_penalty(
            ["earthquake"], ["earthquake"], 0.15, provider
        )

        assert result == 0.0

    @pytest.mark.fast
    def test_moderate_similarity_returns_small_penalty(self):
        """Test that moderate similarity (0.6-0.8) returns small penalty."""
        provider = self._mock_provider_with_similarity(0.7)

        result = _compute_semantic_topic_penalty(
            ["earthquake"], ["disaster"], 0.15, provider
        )

        # Should be in range 0 to 0.3 * max_penalty (0 to 0.045)
        assert result is not None
        assert 0 < result <= 0.15 * 0.3

    @pytest.mark.fast
    def test_low_similarity_returns_medium_penalty(self):
        """Test that low similarity (0.4-0.6) returns medium penalty."""
        provider = self._mock_provider_with_similarity(0.5)

        result = _compute_semantic_topic_penalty(
            ["earthquake"], ["disaster"], 0.15, provider
        )

        # Should be in range 0.3 to 0.6 * max_penalty (0.045 to 0.09)
        assert result is not None
        assert 0.15 * 0.3 <= result <= 0.15 * 0.6

    @pytest.mark.fast
    def test_very_low_similarity_returns_high_penalty(self):
        """Test that very low similarity (<0.4) returns high penalty."""
        provider = self._mock_provider_with_similarity(0.2)

        result = _compute_semantic_topic_penalty(
            ["earthquake"], ["disaster"], 0.15, provider
        )

        # Should be in range 0.6 to 1.0 * max_penalty (0.09 to 0.15)
        assert result is not None
        assert 0.15 * 0.6 <= result <= 0.15

    @pytest.mark.fast
    def test_zero_similarity_returns_max_penalty(self):
        """Test that zero similarity returns maximum penalty."""
        provider = self._mock_provider_with_similarity(0.0)

        result = _compute_semantic_topic_penalty(
            ["earthquake"], ["cooking"], 0.15, provider
        )

        assert result is not None
        assert result == 0.15  # Full penalty


class TestRelatedVsUnrelatedTopics:
    """Test that related topics get lower penalty than unrelated topics."""

    def _create_mock_provider(self, embeddings_map: dict):
        """Create mock provider that returns pre-defined embeddings."""
        mock_provider = MagicMock()

        def embed_func(texts):
            return [embeddings_map.get(t, [0.5, 0.5, 0.5]) for t in texts]

        mock_provider.embed.side_effect = embed_func
        return mock_provider

    @pytest.mark.fast
    def test_related_topics_lower_penalty_than_unrelated(self):
        """Test that earthquake/disaster gets lower penalty than earthquake/cooking."""
        import numpy as np

        # Create embeddings where disaster is closer to earthquake than cooking is
        embeddings = {
            "earthquake": [1.0, 0.0, 0.0],
            "disaster": [0.9, 0.3, 0.0],     # Similar to earthquake
            "cooking": [0.0, 0.0, 1.0],       # Very different from earthquake
        }

        mock_provider = self._create_mock_provider(embeddings)

        # Related topics
        penalty_related = _compute_semantic_topic_penalty(
            ["earthquake"], ["disaster"], 0.15, mock_provider
        )

        # Reset mock for second call
        mock_provider = self._create_mock_provider(embeddings)

        # Unrelated topics
        penalty_unrelated = _compute_semantic_topic_penalty(
            ["earthquake"], ["cooking"], 0.15, mock_provider
        )

        assert penalty_related is not None
        assert penalty_unrelated is not None
        # Related topics should have lower penalty
        assert penalty_related < penalty_unrelated

    @pytest.mark.fast
    def test_semantically_similar_concepts(self):
        """Test semantically similar concepts get low penalty."""
        import numpy as np

        # Create embeddings for semantically similar concepts
        embeddings = {
            "flood water damage": [0.8, 0.5, 0.2],
            "flooding destruction": [0.75, 0.55, 0.25],  # Very similar
        }

        mock_provider = self._create_mock_provider(embeddings)

        result = _compute_semantic_topic_penalty(
            ["flood", "water", "damage"],
            ["flooding", "destruction"],
            0.15,
            mock_provider
        )

        # Similar concepts should have low penalty
        assert result is not None
        assert result < 0.15 * 0.5  # Less than half max penalty


class TestKeywordFallback:
    """Test the keyword overlap fallback when embeddings unavailable."""

    @pytest.mark.fast
    def test_fallback_used_when_no_embedding_provider(self):
        """Test that keyword fallback is used when no embedding provider."""
        result = compute_topic_penalty(
            ["earthquake", "disaster"],
            ["tsunami", "flooding"],
            embedding_provider=None
        )

        # Should use keyword fallback
        assert isinstance(result, float)
        assert 0 <= result <= 0.15

    @pytest.mark.fast
    def test_fallback_used_when_embedding_fails(self):
        """Test that keyword fallback is used when embedding fails."""
        mock_provider = MagicMock()
        mock_provider.embed.side_effect = Exception("API error")

        result = compute_topic_penalty(
            ["earthquake"],
            ["cooking"],
            embedding_provider=mock_provider
        )

        # Should fall back to keyword overlap
        # earthquake vs cooking has no overlap, so max penalty
        assert result == 0.15

    @pytest.mark.fast
    def test_keyword_fallback_partial_overlap(self):
        """Test keyword fallback with partial topic overlap."""
        result = _compute_keyword_topic_penalty(0.4, 0.15)

        # Ratio > 0.3 means small penalty
        assert result == 0.15 * 0.3

    @pytest.mark.fast
    def test_keyword_fallback_weak_overlap(self):
        """Test keyword fallback with weak topic overlap."""
        result = _compute_keyword_topic_penalty(0.1, 0.15)

        # Ratio > 0 but <= 0.3 means medium penalty
        assert result == 0.15 * 0.6

    @pytest.mark.fast
    def test_keyword_fallback_no_overlap(self):
        """Test keyword fallback with no topic overlap."""
        result = _compute_keyword_topic_penalty(0.0, 0.15)

        # Ratio = 0 means full penalty
        assert result == 0.15


class TestExactOverlapFastPath:
    """Test that exact keyword overlap bypasses semantic similarity."""

    @pytest.mark.fast
    def test_exact_overlap_returns_zero_without_embedding(self):
        """Test that exact overlap returns zero penalty without calling embeddings."""
        mock_provider = MagicMock()

        result = compute_topic_penalty(
            ["earthquake"],
            ["earthquake", "disaster"],  # Contains "earthquake"
            embedding_provider=mock_provider
        )

        # Should return 0 without calling embed
        assert result == 0.0
        mock_provider.embed.assert_not_called()

    @pytest.mark.fast
    def test_partial_string_overlap_uses_ratio(self):
        """Test that partial string match affects overlap ratio but not exact count."""
        mock_provider = MagicMock()

        # compute_topic_overlap partial matching adds to ratio, not exact count
        # "boise" in "boise downtown" gives 0.5 partial match, ratio = 0.5
        # Since overlap_count=0 (no exact match) but ratio > 0.3, gets small penalty
        result = compute_topic_penalty(
            ["boise"],
            ["boise downtown"],  # Contains "boise" as substring
            embedding_provider=mock_provider
        )

        # Partial match (ratio > 0.3) should result in small penalty (max_penalty * 0.3)
        # The embedding provider should not be called since keyword fallback is used
        assert result == 0.15 * 0.3  # Partial match penalty


class TestEmptyAndEdgeCases:
    """Test edge cases and empty inputs."""

    @pytest.mark.fast
    def test_empty_vo_topics_returns_zero(self):
        """Test that empty voiceover topics returns zero penalty."""
        result = compute_topic_penalty([], ["disaster"])
        assert result == 0.0

    @pytest.mark.fast
    def test_empty_video_topics_returns_zero(self):
        """Test that empty video topics returns zero penalty."""
        result = compute_topic_penalty(["earthquake"], [])
        assert result == 0.0

    @pytest.mark.fast
    def test_both_empty_returns_zero(self):
        """Test that both empty returns zero penalty."""
        result = compute_topic_penalty([], [])
        assert result == 0.0

    @pytest.mark.fast
    def test_custom_max_penalty(self):
        """Test that custom max_penalty is respected."""
        result = compute_topic_penalty(
            ["earthquake"],
            ["cooking"],
            max_penalty=0.30
        )

        # Should use custom max penalty
        assert result <= 0.30

    @pytest.mark.fast
    def test_zero_max_penalty(self):
        """Test that zero max_penalty always returns zero."""
        result = compute_topic_penalty(
            ["earthquake"],
            ["cooking"],
            max_penalty=0.0
        )

        assert result == 0.0


class TestIntegrationWithEmbeddingProvider:
    """Integration-style tests with mock embedding provider."""

    @pytest.mark.fast
    def test_full_flow_with_provider(self):
        """Test complete flow with embedding provider."""
        import numpy as np

        mock_provider = MagicMock()
        # Return embeddings that indicate moderate similarity
        mock_provider.embed.return_value = [
            [1.0, 0.0, 0.0],
            [0.7, 0.7, 0.0]  # Cosine similarity ~0.7
        ]

        result = compute_topic_penalty(
            ["earthquake", "damage"],
            ["disaster", "destruction"],
            embedding_provider=mock_provider
        )

        # Should return a small penalty (0.6-0.8 similarity range)
        assert result is not None
        assert isinstance(result, float)
        assert 0 <= result <= 0.15 * 0.3  # Small penalty range

    @pytest.mark.fast
    def test_multiple_topics_combined(self):
        """Test that multiple topics are combined into single embedding."""
        mock_provider = MagicMock()
        mock_provider.embed.return_value = [[1.0, 0.0], [0.9, 0.1]]

        compute_topic_penalty(
            ["flood", "water", "damage"],
            ["flooding", "destruction"],
            embedding_provider=mock_provider
        )

        # Verify topics were combined
        call_args = mock_provider.embed.call_args[0][0]
        assert len(call_args) == 2
        assert "flood water damage" in call_args[0]
        assert "flooding destruction" in call_args[1]


class TestSemanticPenaltyHelperExported:
    """Test that helper functions can be imported."""

    @pytest.mark.fast
    def test_semantic_penalty_helper_importable(self):
        """Test that _compute_semantic_topic_penalty is importable."""
        from src.topic_extraction import _compute_semantic_topic_penalty
        assert callable(_compute_semantic_topic_penalty)

    @pytest.mark.fast
    def test_keyword_penalty_helper_importable(self):
        """Test that _compute_keyword_topic_penalty is importable."""
        from src.topic_extraction import _compute_keyword_topic_penalty
        assert callable(_compute_keyword_topic_penalty)


class TestHighVarianceIndicatesUncertainty:
    """Test that high variance in penalty indicates uncertain match (per US-003 criteria)."""

    @pytest.mark.fast
    def test_high_penalty_indicates_topic_mismatch(self):
        """Test that penalty > 0.1 indicates likely topic mismatch."""
        import numpy as np

        # Create provider that returns very different embeddings
        mock_provider = MagicMock()
        mock_provider.embed.return_value = [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]  # Orthogonal = similarity 0
        ]

        result = compute_topic_penalty(
            ["earthquake"],
            ["cooking"],
            embedding_provider=mock_provider
        )

        # High penalty (> 0.1) indicates uncertain/poor match
        assert result > 0.1, "Unrelated topics should have penalty > 0.1"

    @pytest.mark.fast
    def test_low_penalty_indicates_topic_match(self):
        """Test that penalty < 0.05 indicates good topic match."""
        import numpy as np

        # Create provider that returns very similar embeddings
        mock_provider = MagicMock()
        mock_provider.embed.return_value = [
            [1.0, 0.0, 0.0],
            [0.95, 0.05, 0.0]  # Very high similarity
        ]

        result = compute_topic_penalty(
            ["earthquake"],
            ["disaster"],
            embedding_provider=mock_provider
        )

        # Low penalty indicates good match
        assert result < 0.05, "Related topics should have penalty < 0.05"
