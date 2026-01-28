"""
Tests for multi-modal similarity weighting.

Tests the compute_multimodal_score() function and related helper functions
in src/matching/scoring.py, as well as integration with TieredMatcher.

US-010: Add multi-modal similarity weighting
"""

import pytest
from unittest.mock import MagicMock, patch
import math

from src.matching.scoring import (
    compute_multimodal_score,
    calculate_keyword_overlap_score,
    calculate_entity_match_score,
    calculate_visual_description_score,
    DEFAULT_MULTIMODAL_WEIGHTS,
)


class TestDefaultWeights:
    """Tests for default multimodal weights."""

    @pytest.mark.fast
    def test_default_weights_exist(self):
        """Default weights dictionary exists with all required keys."""
        assert DEFAULT_MULTIMODAL_WEIGHTS is not None
        assert 'text_embedding' in DEFAULT_MULTIMODAL_WEIGHTS
        assert 'keyword_overlap' in DEFAULT_MULTIMODAL_WEIGHTS
        assert 'entity_match' in DEFAULT_MULTIMODAL_WEIGHTS
        assert 'visual_description' in DEFAULT_MULTIMODAL_WEIGHTS

    @pytest.mark.fast
    def test_default_weights_sum_to_one(self):
        """Default weights sum to 1.0."""
        weight_sum = sum(DEFAULT_MULTIMODAL_WEIGHTS.values())
        assert abs(weight_sum - 1.0) < 0.001

    @pytest.mark.fast
    def test_default_weights_values(self):
        """Default weights have expected values."""
        assert DEFAULT_MULTIMODAL_WEIGHTS['text_embedding'] == 0.40
        assert DEFAULT_MULTIMODAL_WEIGHTS['keyword_overlap'] == 0.25
        assert DEFAULT_MULTIMODAL_WEIGHTS['entity_match'] == 0.20
        assert DEFAULT_MULTIMODAL_WEIGHTS['visual_description'] == 0.15


class TestComputeMultimodalScoreBasic:
    """Basic tests for compute_multimodal_score function."""

    @pytest.mark.fast
    def test_disabled_returns_embedding_similarity(self):
        """When disabled, returns embedding similarity unchanged."""
        result, reason, components = compute_multimodal_score(
            embedding_similarity=0.85,
            keyword_overlap_score=0.6,
            entity_match_score=0.5,
            visual_description_score=0.4,
            multimodal_enabled=False
        )
        assert result == 0.85
        assert reason == "multimodal_disabled"
        assert components['weights_used'] is None

    @pytest.mark.fast
    def test_enabled_combines_signals(self):
        """When enabled, combines all signals with weights."""
        result, reason, components = compute_multimodal_score(
            embedding_similarity=0.8,
            keyword_overlap_score=0.6,
            entity_match_score=0.5,
            visual_description_score=0.4,
            multimodal_enabled=True
        )
        # Expected: 0.8*0.4 + 0.6*0.25 + 0.5*0.2 + 0.4*0.15
        # = 0.32 + 0.15 + 0.1 + 0.06 = 0.63
        assert abs(result - 0.63) < 0.01
        assert "multimodal" in reason

    @pytest.mark.fast
    def test_zero_scores_returns_zero(self):
        """All zero scores returns zero."""
        result, reason, components = compute_multimodal_score(
            embedding_similarity=0.0,
            keyword_overlap_score=0.0,
            entity_match_score=0.0,
            visual_description_score=0.0,
            multimodal_enabled=True
        )
        assert result == 0.0

    @pytest.mark.fast
    def test_perfect_scores_returns_one(self):
        """All perfect scores returns 1.0."""
        result, reason, components = compute_multimodal_score(
            embedding_similarity=1.0,
            keyword_overlap_score=1.0,
            entity_match_score=1.0,
            visual_description_score=1.0,
            multimodal_enabled=True
        )
        assert abs(result - 1.0) < 0.01


class TestWeightedContributions:
    """Tests for weighted contribution calculations."""

    @pytest.mark.fast
    def test_embedding_only_contribution(self):
        """Only embedding similarity contributes."""
        result, reason, components = compute_multimodal_score(
            embedding_similarity=1.0,
            keyword_overlap_score=0.0,
            entity_match_score=0.0,
            visual_description_score=0.0,
            multimodal_enabled=True
        )
        # 1.0 * 0.4 = 0.4
        assert abs(result - 0.4) < 0.01
        assert components['embedding_contribution'] == 0.4

    @pytest.mark.fast
    def test_keyword_only_contribution(self):
        """Only keyword overlap contributes."""
        result, reason, components = compute_multimodal_score(
            embedding_similarity=0.0,
            keyword_overlap_score=1.0,
            entity_match_score=0.0,
            visual_description_score=0.0,
            multimodal_enabled=True
        )
        # 1.0 * 0.25 = 0.25
        assert abs(result - 0.25) < 0.01
        assert components['keyword_contribution'] == 0.25

    @pytest.mark.fast
    def test_entity_only_contribution(self):
        """Only entity match contributes."""
        result, reason, components = compute_multimodal_score(
            embedding_similarity=0.0,
            keyword_overlap_score=0.0,
            entity_match_score=1.0,
            visual_description_score=0.0,
            multimodal_enabled=True
        )
        # 1.0 * 0.2 = 0.2
        assert abs(result - 0.2) < 0.01
        assert components['entity_contribution'] == 0.2

    @pytest.mark.fast
    def test_visual_only_contribution(self):
        """Only visual description contributes."""
        result, reason, components = compute_multimodal_score(
            embedding_similarity=0.0,
            keyword_overlap_score=0.0,
            entity_match_score=0.0,
            visual_description_score=1.0,
            multimodal_enabled=True
        )
        # 1.0 * 0.15 = 0.15
        assert abs(result - 0.15) < 0.01
        assert components['visual_contribution'] == 0.15


class TestCustomWeights:
    """Tests for custom weight configurations."""

    @pytest.mark.fast
    def test_custom_weights_used(self):
        """Custom weights are used instead of defaults."""
        custom_weights = {
            'text_embedding': 0.5,
            'keyword_overlap': 0.3,
            'entity_match': 0.1,
            'visual_description': 0.1
        }
        result, reason, components = compute_multimodal_score(
            embedding_similarity=1.0,
            keyword_overlap_score=0.0,
            entity_match_score=0.0,
            visual_description_score=0.0,
            weights=custom_weights,
            multimodal_enabled=True
        )
        # With custom weights, embedding only = 1.0 * 0.5 = 0.5
        assert abs(result - 0.5) < 0.01

    @pytest.mark.fast
    def test_weights_normalized_when_not_sum_to_one(self):
        """Weights are normalized when they don't sum to 1.0."""
        bad_weights = {
            'text_embedding': 0.8,
            'keyword_overlap': 0.5,
            'entity_match': 0.4,
            'visual_description': 0.3
        }
        # Sum = 2.0, should be normalized
        result, reason, components = compute_multimodal_score(
            embedding_similarity=1.0,
            keyword_overlap_score=1.0,
            entity_match_score=1.0,
            visual_description_score=1.0,
            weights=bad_weights,
            multimodal_enabled=True
        )
        # All 1.0 inputs should give ~1.0 output after normalization
        assert abs(result - 1.0) < 0.01


class TestInputClamping:
    """Tests for input value clamping."""

    @pytest.mark.fast
    def test_negative_inputs_clamped_to_zero(self):
        """Negative input scores are clamped to 0."""
        result, reason, components = compute_multimodal_score(
            embedding_similarity=-0.5,
            keyword_overlap_score=-0.3,
            entity_match_score=-0.2,
            visual_description_score=-0.1,
            multimodal_enabled=True
        )
        assert result == 0.0
        assert components['embedding_similarity'] == 0.0

    @pytest.mark.fast
    def test_oversized_inputs_clamped_to_one(self):
        """Scores above 1.0 are clamped to 1.0."""
        result, reason, components = compute_multimodal_score(
            embedding_similarity=1.5,
            keyword_overlap_score=1.2,
            entity_match_score=1.3,
            visual_description_score=1.1,
            multimodal_enabled=True
        )
        assert abs(result - 1.0) < 0.01
        assert components['embedding_similarity'] == 1.0


class TestReasonString:
    """Tests for reason string formatting."""

    @pytest.mark.fast
    def test_reason_includes_components(self):
        """Reason string includes component contributions."""
        result, reason, components = compute_multimodal_score(
            embedding_similarity=0.8,
            keyword_overlap_score=0.6,
            entity_match_score=0.5,
            visual_description_score=0.4,
            multimodal_enabled=True
        )
        assert "emb:" in reason
        assert "kw:" in reason
        assert "ent:" in reason
        assert "vis:" in reason

    @pytest.mark.fast
    def test_reason_shows_zero_components(self):
        """Reason string omits zero-score components."""
        result, reason, components = compute_multimodal_score(
            embedding_similarity=0.8,
            keyword_overlap_score=0.0,
            entity_match_score=0.0,
            visual_description_score=0.0,
            multimodal_enabled=True
        )
        assert "emb:" in reason
        assert "kw:" not in reason
        assert "ent:" not in reason
        assert "vis:" not in reason


class TestKeywordOverlapScore:
    """Tests for calculate_keyword_overlap_score function."""

    @pytest.mark.fast
    def test_no_vo_keywords_returns_zero(self):
        """Empty voiceover keywords returns zero."""
        score, matched = calculate_keyword_overlap_score([], ["keyword1"])
        assert score == 0.0
        assert matched == []

    @pytest.mark.fast
    def test_no_video_keywords_returns_zero(self):
        """Empty video keywords returns zero."""
        score, matched = calculate_keyword_overlap_score(["keyword1"], [])
        assert score == 0.0
        assert matched == []

    @pytest.mark.fast
    def test_no_overlap_returns_zero(self):
        """No overlap returns zero."""
        score, matched = calculate_keyword_overlap_score(
            ["apple", "banana"],
            ["cherry", "date"]
        )
        assert score == 0.0
        assert matched == []

    @pytest.mark.fast
    def test_one_match_returns_035(self):
        """One keyword match returns 0.35."""
        score, matched = calculate_keyword_overlap_score(
            ["apple", "banana"],
            ["apple", "cherry"]
        )
        assert abs(score - 0.35) < 0.01
        assert "apple" in [m.lower() for m in matched]

    @pytest.mark.fast
    def test_two_matches_returns_055(self):
        """Two keyword matches returns 0.55."""
        score, matched = calculate_keyword_overlap_score(
            ["apple", "banana", "cherry"],
            ["apple", "banana", "date"]
        )
        assert abs(score - 0.55) < 0.01
        assert len(matched) == 2

    @pytest.mark.fast
    def test_three_matches_returns_075(self):
        """Three keyword matches returns 0.75."""
        score, matched = calculate_keyword_overlap_score(
            ["apple", "banana", "cherry", "date"],
            ["apple", "banana", "cherry", "fig"]
        )
        assert abs(score - 0.75) < 0.01
        assert len(matched) == 3

    @pytest.mark.fast
    def test_four_matches_returns_090(self):
        """Four keyword matches returns 0.9."""
        score, matched = calculate_keyword_overlap_score(
            ["apple", "banana", "cherry", "date", "fig"],
            ["apple", "banana", "cherry", "date", "grape"]
        )
        assert abs(score - 0.9) < 0.01
        assert len(matched) == 4

    @pytest.mark.fast
    def test_five_plus_matches_returns_100(self):
        """Five or more keyword matches returns 1.0."""
        score, matched = calculate_keyword_overlap_score(
            ["apple", "banana", "cherry", "date", "fig", "grape"],
            ["apple", "banana", "cherry", "date", "fig", "kiwi"]
        )
        assert score == 1.0
        assert len(matched) == 5

    @pytest.mark.fast
    def test_case_insensitive_matching(self):
        """Keyword matching is case-insensitive."""
        score, matched = calculate_keyword_overlap_score(
            ["Apple", "BANANA"],
            ["apple", "banana"]
        )
        assert abs(score - 0.55) < 0.01
        assert len(matched) == 2


class TestEntityMatchScore:
    """Tests for calculate_entity_match_score function."""

    @pytest.mark.fast
    def test_no_vo_entities_returns_zero(self):
        """Empty voiceover entities returns zero."""
        score, matched = calculate_entity_match_score([], ["New York"])
        assert score == 0.0
        assert matched == []

    @pytest.mark.fast
    def test_no_video_entities_returns_zero(self):
        """Empty video entities returns zero."""
        score, matched = calculate_entity_match_score(["New York"], [])
        assert score == 0.0
        assert matched == []

    @pytest.mark.fast
    def test_no_overlap_returns_zero(self):
        """No entity overlap returns zero."""
        score, matched = calculate_entity_match_score(
            ["New York", "Paris"],
            ["London", "Tokyo"]
        )
        assert score == 0.0
        assert matched == []

    @pytest.mark.fast
    def test_one_entity_match_returns_060(self):
        """One entity match returns 0.6."""
        score, matched = calculate_entity_match_score(
            ["New York", "Paris"],
            ["New York", "London"]
        )
        assert abs(score - 0.6) < 0.01
        assert len(matched) == 1

    @pytest.mark.fast
    def test_two_entity_matches_returns_080(self):
        """Two entity matches returns 0.8."""
        score, matched = calculate_entity_match_score(
            ["New York", "Paris", "Tokyo"],
            ["New York", "Paris", "London"]
        )
        assert abs(score - 0.8) < 0.01
        assert len(matched) == 2

    @pytest.mark.fast
    def test_three_plus_entity_matches_returns_100(self):
        """Three or more entity matches returns 1.0."""
        score, matched = calculate_entity_match_score(
            ["New York", "Paris", "Tokyo", "London"],
            ["New York", "Paris", "Tokyo", "Berlin"]
        )
        assert score == 1.0
        assert len(matched) == 3

    @pytest.mark.fast
    def test_case_insensitive_entity_matching(self):
        """Entity matching is case-insensitive."""
        score, matched = calculate_entity_match_score(
            ["NEW YORK", "paris"],
            ["New York", "Paris"]
        )
        assert abs(score - 0.8) < 0.01

    @pytest.mark.fast
    def test_short_entities_filtered(self):
        """Very short entities (< 2 chars) are filtered."""
        score, matched = calculate_entity_match_score(
            ["A", "B", "New York"],
            ["A", "B", "New York"]
        )
        # Only "New York" should count
        assert abs(score - 0.6) < 0.01
        assert len(matched) == 1


class TestVisualDescriptionScore:
    """Tests for calculate_visual_description_score function."""

    @pytest.mark.fast
    def test_empty_voiceover_returns_zero(self):
        """Empty voiceover text returns zero."""
        score = calculate_visual_description_score("", "description text")
        assert score == 0.0

    @pytest.mark.fast
    def test_no_description_or_keywords_returns_zero(self):
        """No description or keywords returns zero."""
        score = calculate_visual_description_score("test voiceover", "", None)
        assert score == 0.0

    @pytest.mark.fast
    def test_matching_description_words(self):
        """Matching words in description gives positive score."""
        score = calculate_visual_description_score(
            "Beautiful sunset over the ocean waves",
            "ocean waves sunset beach scene"
        )
        assert score > 0
        assert score <= 1.0

    @pytest.mark.fast
    def test_matching_visual_keywords(self):
        """Matching visual keywords gives positive score."""
        score = calculate_visual_description_score(
            "The Eiffel Tower stands tall in Paris",
            "",
            ["Eiffel Tower", "Paris", "France"]
        )
        assert score > 0
        assert score <= 1.0

    @pytest.mark.fast
    def test_no_matching_content(self):
        """No matching content returns zero or very low score."""
        score = calculate_visual_description_score(
            "Computer programming tutorial",
            "Beautiful beach sunset"
        )
        # Should be very low due to no word overlap
        assert score < 0.3

    @pytest.mark.fast
    def test_short_words_filtered(self):
        """Short words (< 4 chars) are filtered out."""
        score = calculate_visual_description_score(
            "a the is of",  # All short/common words
            "a the is of"
        )
        # No significant words to match
        assert score == 0.0


class TestComponentScoresDictionary:
    """Tests for component scores dictionary returned by compute_multimodal_score."""

    @pytest.mark.fast
    def test_component_scores_contains_all_fields(self):
        """Component scores dict contains all expected fields."""
        _, _, components = compute_multimodal_score(
            embedding_similarity=0.8,
            keyword_overlap_score=0.6,
            entity_match_score=0.5,
            visual_description_score=0.4,
            multimodal_enabled=True
        )

        assert 'embedding_similarity' in components
        assert 'keyword_overlap' in components
        assert 'entity_match' in components
        assert 'visual_description' in components
        assert 'embedding_contribution' in components
        assert 'keyword_contribution' in components
        assert 'entity_contribution' in components
        assert 'visual_contribution' in components
        assert 'weights_used' in components

    @pytest.mark.fast
    def test_component_scores_values_are_clamped(self):
        """Component scores in dict are clamped versions of inputs."""
        _, _, components = compute_multimodal_score(
            embedding_similarity=1.5,  # Should be clamped to 1.0
            keyword_overlap_score=-0.2,  # Should be clamped to 0.0
            entity_match_score=0.5,
            visual_description_score=0.4,
            multimodal_enabled=True
        )

        assert components['embedding_similarity'] == 1.0
        assert components['keyword_overlap'] == 0.0


class TestMultimodalIntegration:
    """Integration tests with TieredMatcher."""

    @pytest.mark.fast
    def test_tiered_matcher_uses_multimodal(self):
        """TieredMatcher._compute_multimodal_confidence is callable."""
        from src.matching.tiered_matcher import TieredMatcher

        # Create a mock config
        mock_config = MagicMock()
        mock_config.matching.multimodal_enabled = True
        mock_config.matching.multimodal_weights = None
        mock_config.matching.gemini_model = 'test'
        mock_config.matching.primary_provider = 'gemini'
        mock_config.matching.secondary_provider = None
        mock_config.matching.use_local_for_review = False
        mock_config.matching.max_clip_reuse = 2
        mock_config.matching.reuse_penalty = 0.5
        mock_config.matching.max_source_file_reuse = 0
        mock_config.matching.source_file_penalty = 0.05
        mock_config.matching.face_preference = 'neutral'
        mock_config.matching.location_matching = None
        mock_config.gemini_api_key = None
        mock_config.anthropic_api_key = None

        matcher = TieredMatcher(config=mock_config)

        # Create mock segments
        vo_seg = MagicMock()
        vo_seg.text = "Test voiceover about Paris"
        vo_seg.keywords = ["Paris", "travel"]
        vo_seg.entities = [{"text": "Paris"}]

        video_seg = MagicMock()
        video_seg.text = "Paris travel footage"
        video_seg.keywords = ["Paris", "France"]
        video_seg.entities = [{"text": "Paris"}]

        # Call the method
        conf, reason, components = matcher._compute_multimodal_confidence(
            vo_seg, video_seg, 0.85, None
        )

        assert 0 <= conf <= 1.0
        assert "multimodal" in reason

    @pytest.mark.fast
    def test_tiered_matcher_disabled_multimodal(self):
        """TieredMatcher returns embedding sim when multimodal disabled."""
        from src.matching.tiered_matcher import TieredMatcher

        mock_config = MagicMock()
        mock_config.matching.multimodal_enabled = False
        mock_config.matching.multimodal_weights = None
        mock_config.matching.gemini_model = 'test'
        mock_config.matching.primary_provider = 'gemini'
        mock_config.matching.secondary_provider = None
        mock_config.matching.use_local_for_review = False
        mock_config.matching.max_clip_reuse = 2
        mock_config.matching.reuse_penalty = 0.5
        mock_config.matching.max_source_file_reuse = 0
        mock_config.matching.source_file_penalty = 0.05
        mock_config.matching.face_preference = 'neutral'
        mock_config.matching.location_matching = None
        mock_config.gemini_api_key = None
        mock_config.anthropic_api_key = None

        matcher = TieredMatcher(config=mock_config)

        vo_seg = MagicMock()
        vo_seg.text = "Test"
        vo_seg.keywords = []
        vo_seg.entities = []

        video_seg = MagicMock()
        video_seg.text = "Test"
        video_seg.keywords = []
        video_seg.entities = []

        conf, reason, components = matcher._compute_multimodal_confidence(
            vo_seg, video_seg, 0.85, None
        )

        assert conf == 0.85
        assert reason == "multimodal_disabled"


class TestEdgeCases:
    """Edge case tests."""

    @pytest.mark.fast
    def test_none_weights_uses_defaults(self):
        """None weights parameter uses default weights."""
        result, reason, components = compute_multimodal_score(
            embedding_similarity=0.8,
            keyword_overlap_score=0.6,
            entity_match_score=0.5,
            visual_description_score=0.4,
            weights=None,
            multimodal_enabled=True
        )
        assert components['weights_used'] == DEFAULT_MULTIMODAL_WEIGHTS

    @pytest.mark.fast
    def test_empty_weights_dict_uses_defaults(self):
        """Empty weights dict falls back to defaults for missing keys."""
        result, reason, components = compute_multimodal_score(
            embedding_similarity=1.0,
            keyword_overlap_score=1.0,
            entity_match_score=1.0,
            visual_description_score=1.0,
            weights={},  # Empty dict
            multimodal_enabled=True
        )
        # With empty dict, all contributions will be 0 (no keys found)
        # This is edge case behavior - weights.get() returns default 0.4, etc.
        assert result > 0  # Should still work with defaults in .get()

    @pytest.mark.fast
    def test_partial_weights_dict(self):
        """Partial weights dict uses defaults for missing keys."""
        partial_weights = {
            'text_embedding': 0.6
        }
        result, reason, components = compute_multimodal_score(
            embedding_similarity=1.0,
            keyword_overlap_score=0.0,
            entity_match_score=0.0,
            visual_description_score=0.0,
            weights=partial_weights,
            multimodal_enabled=True
        )
        # Should use 0.6 for embedding (custom) and defaults for others
        # But since weight sum != 1.0, it gets normalized
        assert result > 0

    @pytest.mark.fast
    def test_extremely_small_weights(self):
        """Very small weights still produce valid results."""
        small_weights = {
            'text_embedding': 0.001,
            'keyword_overlap': 0.001,
            'entity_match': 0.001,
            'visual_description': 0.001
        }
        result, reason, components = compute_multimodal_score(
            embedding_similarity=1.0,
            keyword_overlap_score=1.0,
            entity_match_score=1.0,
            visual_description_score=1.0,
            weights=small_weights,
            multimodal_enabled=True
        )
        # Weights sum to 0.004, will be normalized to 1.0
        # All inputs 1.0, so result should be ~1.0
        assert abs(result - 1.0) < 0.01


class TestFormulaVerification:
    """Tests to verify the formula is applied correctly."""

    @pytest.mark.fast
    def test_exact_formula_calculation(self):
        """Verify exact formula: sum(score_i * weight_i)."""
        # Known inputs
        emb = 0.8
        kw = 0.6
        ent = 0.5
        vis = 0.4

        # Expected calculation with default weights
        expected = (
            emb * 0.40 +  # 0.32
            kw * 0.25 +   # 0.15
            ent * 0.20 +  # 0.10
            vis * 0.15    # 0.06
        )  # = 0.63

        result, _, _ = compute_multimodal_score(
            embedding_similarity=emb,
            keyword_overlap_score=kw,
            entity_match_score=ent,
            visual_description_score=vis,
            multimodal_enabled=True
        )

        assert abs(result - expected) < 0.001

    @pytest.mark.fast
    def test_contribution_sums_match_total(self):
        """Individual contributions sum to total score."""
        result, _, components = compute_multimodal_score(
            embedding_similarity=0.9,
            keyword_overlap_score=0.7,
            entity_match_score=0.6,
            visual_description_score=0.5,
            multimodal_enabled=True
        )

        contrib_sum = (
            components['embedding_contribution'] +
            components['keyword_contribution'] +
            components['entity_contribution'] +
            components['visual_contribution']
        )

        assert abs(result - contrib_sum) < 0.001
