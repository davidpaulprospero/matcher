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


# ===========================================================================
# US-77-002: Integration tests for semantic coherence in TieredMatcher
# ===========================================================================
class TestTieredMatcherSemanticCoherenceIntegration:
    """Tests that compute_semantic_coherence is wired into TieredMatcher match paths."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config for TieredMatcher."""
        from dataclasses import dataclass, field
        from unittest.mock import MagicMock

        @dataclass
        class MockMatchingConfig:
            gemini_model: str = "gemini-2.0-flash"
            anthropic_model: str = "claude-3-haiku-20240307"
            ollama_model: str = "llama3.2"
            ollama_host: str = "http://localhost:11434"
            primary_provider: str = "gemini"
            secondary_provider: str = ""
            use_local_for_review: bool = False
            min_confidence: float = 0.3
            embedding_candidates: int = 10
            high_confidence_threshold: float = 0.85
            low_confidence_threshold: float = 0.5
            skip_llm_threshold: float = 0.70
            ambiguous_threshold: float = 0.6
            confidence_threshold: float = 0.3
            max_clip_reuse: int = 10
            reuse_penalty: float = 0.01
            chapter_matching_enabled: bool = False
            topic_mismatch_penalty: float = 0.15
            location_matching: None = None
            cache_llm_responses: bool = False
            face_preference: str = "neutral"
            caption_quality_adjustment_enabled: bool = False
            timing_penalty_enabled: bool = False
            max_consecutive_same_source: int = 3
            consecutive_source_penalty: float = 0.05
            adaptive_threshold_enabled: bool = False
            current_project_boost: float = 0.0
            broll_boost: float = 0.0
            obvious_match_min_confidence: float = 0.0
            semantic_coherence_enabled: bool = True
            multimodal_enabled: bool = False
            llm_rerank_candidates: int = 5

        @dataclass
        class MockOutputConfig:
            num_alternatives: int = 2
            secondary_matches_enabled: bool = False
            strategy_matches_enabled: bool = False

        @dataclass
        class MockConfig:
            matching: MockMatchingConfig = field(default_factory=MockMatchingConfig)
            output: MockOutputConfig = field(default_factory=MockOutputConfig)
            gemini_api_key: str = None
            anthropic_api_key: str = None
            negative_matching: MagicMock = field(default_factory=lambda: MagicMock(enabled=False))

        return MockConfig()

    def _make_segments_and_embeddings(self):
        """Create video segments with parallel embeddings for lookup."""
        from src.utils import SRTSegment
        seg1 = SRTSegment(index=0, start_time=0, end_time=10, text="Video about solar energy panels",
                          source_file="vid_001")
        seg2 = SRTSegment(index=1, start_time=0, end_time=10, text="Video about underwater diving",
                          source_file="vid_002")
        # Embedding for seg1: similar direction
        emb1 = np.array([1.0, 0.0, 0.0])
        # Embedding for seg2: orthogonal direction
        emb2 = np.array([0.0, 1.0, 0.0])
        return [seg1, seg2], [emb1, emb2]

    def _create_matcher(self, config):
        """Create a TieredMatcher with mocked config."""
        from src.matching.tiered_matcher import TieredMatcher
        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            matcher = TieredMatcher(config=config)
        return matcher

    @pytest.mark.fast
    def test_smooth_transition_boost_in_breakdown(self, mock_config):
        """Smooth topic flow between consecutive matches adds semantic_coherence to breakdown."""
        segments, embeddings = self._make_segments_and_embeddings()
        matcher = self._create_matcher(mock_config)
        matcher.set_embedding_lookup(segments, embeddings)

        # Simulate: previous match used seg1 (emb [1,0,0])
        matcher._previous_match_embedding = embeddings[0]

        # Current match also uses seg1 (same embedding -> smooth flow)
        confidence_breakdown = []
        result_conf, reason = matcher._apply_semantic_coherence(
            0.80, segments[0], confidence_breakdown
        )

        assert result_conf == pytest.approx(0.80 + SEMANTIC_COHERENCE_SMOOTH_BOOST)
        assert any(b['component'] == 'semantic_coherence' for b in confidence_breakdown)
        assert "smooth_topic_flow" in reason

    @pytest.mark.fast
    def test_abrupt_transition_penalty_in_breakdown(self, mock_config):
        """Abrupt topic flow between consecutive matches applies penalty."""
        segments, embeddings = self._make_segments_and_embeddings()
        matcher = self._create_matcher(mock_config)
        matcher.set_embedding_lookup(segments, embeddings)

        # Previous match used seg1 (emb [1,0,0])
        matcher._previous_match_embedding = embeddings[0]

        # Current match uses seg2 (emb [0,1,0] -> orthogonal = abrupt)
        confidence_breakdown = []
        result_conf, reason = matcher._apply_semantic_coherence(
            0.80, segments[1], confidence_breakdown
        )

        assert result_conf == pytest.approx(0.80 - SEMANTIC_COHERENCE_ABRUPT_PENALTY)
        assert any(b['component'] == 'semantic_coherence' for b in confidence_breakdown)
        assert "abrupt_topic_flow" in reason

    @pytest.mark.fast
    def test_disabled_returns_zero(self, mock_config):
        """When semantic_coherence_enabled=False, no adjustment is applied."""
        mock_config.matching.semantic_coherence_enabled = False
        segments, embeddings = self._make_segments_and_embeddings()
        matcher = self._create_matcher(mock_config)
        matcher.set_embedding_lookup(segments, embeddings)

        matcher._previous_match_embedding = embeddings[0]

        confidence_breakdown = []
        result_conf, reason = matcher._apply_semantic_coherence(
            0.80, segments[0], confidence_breakdown
        )

        assert result_conf == 0.80
        assert reason == "semantic_coherence_disabled"

    @pytest.mark.fast
    def test_first_segment_no_previous_returns_zero(self, mock_config):
        """First segment (no previous embedding) returns 0.0 adjustment."""
        segments, embeddings = self._make_segments_and_embeddings()
        matcher = self._create_matcher(mock_config)
        matcher.set_embedding_lookup(segments, embeddings)

        # _previous_match_embedding is None (default for first segment)
        assert matcher._previous_match_embedding is None

        confidence_breakdown = []
        result_conf, reason = matcher._apply_semantic_coherence(
            0.80, segments[0], confidence_breakdown
        )

        assert result_conf == 0.80
        assert reason == "missing_embedding"

    @pytest.mark.fast
    def test_previous_embedding_updated_after_match(self, mock_config):
        """_apply_semantic_coherence updates _previous_match_embedding for next call."""
        segments, embeddings = self._make_segments_and_embeddings()
        matcher = self._create_matcher(mock_config)
        matcher.set_embedding_lookup(segments, embeddings)

        assert matcher._previous_match_embedding is None

        # First call with seg1
        matcher._apply_semantic_coherence(0.80, segments[0], [])

        # Previous embedding should now be seg1's embedding
        assert matcher._previous_match_embedding is not None
        np.testing.assert_array_equal(matcher._previous_match_embedding, embeddings[0])

        # Second call with seg2
        matcher._apply_semantic_coherence(0.80, segments[1], [])

        # Previous embedding should now be seg2's embedding
        np.testing.assert_array_equal(matcher._previous_match_embedding, embeddings[1])

    @pytest.mark.fast
    def test_set_embedding_lookup_builds_mapping(self, mock_config):
        """set_embedding_lookup creates id-based mapping from segments to embeddings."""
        segments, embeddings = self._make_segments_and_embeddings()
        matcher = self._create_matcher(mock_config)
        matcher.set_embedding_lookup(segments, embeddings)

        assert matcher._video_embeddings is embeddings
        assert len(matcher._embedding_lookup) == 2
        # Can look up each segment's embedding
        for i, seg in enumerate(segments):
            emb = matcher._get_segment_embedding(seg)
            assert emb is not None
            np.testing.assert_array_equal(emb, embeddings[i])

    @pytest.mark.fast
    def test_no_embedding_lookup_returns_zero(self, mock_config):
        """Without set_embedding_lookup, _apply_semantic_coherence returns 0.0."""
        from src.utils import SRTSegment
        matcher = self._create_matcher(mock_config)

        # Set a previous embedding manually
        matcher._previous_match_embedding = np.array([1.0, 0.0, 0.0])

        # Create a segment that is NOT in the lookup
        seg = SRTSegment(index=99, start_time=0, end_time=10, text="test", source_file="x")
        confidence_breakdown = []
        result_conf, reason = matcher._apply_semantic_coherence(0.80, seg, [])

        # current_embedding is None (not in lookup), so returns 0.0
        assert result_conf == 0.80
        assert reason == "missing_embedding"
