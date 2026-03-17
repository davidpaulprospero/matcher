"""Tests for validate() coverage of context_enrichment and chapter_grouping constraints (US-80-006).

Covers:
- AC1: chapter_topic_match_boost is list of exactly 2 floats [min, max] with min <= max
- AC2: chapter_topic_mismatch_penalty must be negative
- AC3: context_enrichment.max_description_length must be positive
- AC4: Cross-section: title_enriched_embeddings=true requires embedding provider
- AC5: Specific test: chapter_topic_match_boost=[0.10, 0.05] returns error (min > max)
"""

import pytest

from src.config.base import Config


@pytest.mark.fast
class TestChapterGroupingConstraints:
    """Test _validate_constraints covers chapter_grouping fields."""

    def _make_config_with_chapter_grouping(self, **overrides):
        """Create Config and mutate chapter_grouping fields post-construction."""
        config = Config()
        for key, value in overrides.items():
            setattr(config.matching.chapter_grouping, key, value)
        return config

    def test_chapter_topic_match_boost_min_gt_max_returns_error(self):
        """AC5: validate() returns error when chapter_topic_match_boost=[0.10, 0.05] (min > max)."""
        config = self._make_config_with_chapter_grouping(
            chapter_topic_match_boost=[0.10, 0.05]
        )
        errors = config._validate_constraints()
        assert any(
            "chapter_topic_match_boost" in e and "min" in e and "max" in e
            for e in errors
        ), f"Expected chapter_topic_match_boost min>max error, got: {errors}"

    def test_chapter_topic_match_boost_valid_no_error(self):
        """Valid [min, max] with min <= max passes."""
        config = self._make_config_with_chapter_grouping(
            chapter_topic_match_boost=[0.05, 0.15]
        )
        errors = config._validate_constraints()
        assert not any("chapter_topic_match_boost" in e for e in errors)

    def test_chapter_topic_match_boost_equal_min_max_valid(self):
        """Equal min and max is valid."""
        config = self._make_config_with_chapter_grouping(
            chapter_topic_match_boost=[0.10, 0.10]
        )
        errors = config._validate_constraints()
        assert not any("chapter_topic_match_boost" in e for e in errors)

    def test_chapter_topic_match_boost_not_list_returns_error(self):
        """Non-list value returns error."""
        config = self._make_config_with_chapter_grouping(
            chapter_topic_match_boost=0.10
        )
        errors = config._validate_constraints()
        assert any("chapter_topic_match_boost" in e for e in errors)

    def test_chapter_topic_match_boost_wrong_length_returns_error(self):
        """List with != 2 elements returns error."""
        config = self._make_config_with_chapter_grouping(
            chapter_topic_match_boost=[0.05, 0.10, 0.15]
        )
        errors = config._validate_constraints()
        assert any("chapter_topic_match_boost" in e for e in errors)

    def test_chapter_topic_mismatch_penalty_positive_returns_error(self):
        """AC2: validate() checks chapter_topic_mismatch_penalty is negative."""
        config = self._make_config_with_chapter_grouping(
            chapter_topic_mismatch_penalty=0.05
        )
        errors = config._validate_constraints()
        assert any(
            "chapter_topic_mismatch_penalty" in e and "negative" in e
            for e in errors
        ), f"Expected mismatch_penalty positive error, got: {errors}"

    def test_chapter_topic_mismatch_penalty_zero_returns_error(self):
        """Zero is not negative, should return error."""
        config = self._make_config_with_chapter_grouping(
            chapter_topic_mismatch_penalty=0.0
        )
        errors = config._validate_constraints()
        assert any("chapter_topic_mismatch_penalty" in e for e in errors)

    def test_chapter_topic_mismatch_penalty_negative_valid(self):
        """Negative penalty passes validation."""
        config = self._make_config_with_chapter_grouping(
            chapter_topic_mismatch_penalty=-0.10
        )
        errors = config._validate_constraints()
        assert not any("chapter_topic_mismatch_penalty" in e for e in errors)

    def test_coherence_threshold_less_than_min_diversity_returns_error(self):
        """coherence_penalty_threshold must be >= min_source_diversity."""
        config = self._make_config_with_chapter_grouping(
            coherence_penalty_threshold=3,
            min_source_diversity=5
        )
        errors = config._validate_constraints()
        assert any(
            "coherence_penalty_threshold" in e and "min_source_diversity" in e
            for e in errors
        ), f"Expected coherence_threshold < min_diversity error, got: {errors}"

    def test_coherence_threshold_equal_min_diversity_valid(self):
        """coherence_penalty_threshold = min_source_diversity is valid."""
        config = self._make_config_with_chapter_grouping(
            coherence_penalty_threshold=5,
            min_source_diversity=5
        )
        errors = config._validate_constraints()
        assert not any("coherence_penalty_threshold" in e for e in errors)

    def test_coherence_threshold_greater_than_min_diversity_valid(self):
        """coherence_penalty_threshold > min_source_diversity is valid."""
        config = self._make_config_with_chapter_grouping(
            coherence_penalty_threshold=7,
            min_source_diversity=3
        )
        errors = config._validate_constraints()
        assert not any("coherence_penalty_threshold" in e for e in errors)

    def test_multi_chapter_assignment_strategy_invalid_returns_error(self):
        """multi_chapter_assignment_strategy must be one of valid options."""
        config = self._make_config_with_chapter_grouping(
            multi_chapter_assignment_strategy="invalid_option"
        )
        errors = config._validate_constraints()
        assert any(
            "multi_chapter_assignment_strategy" in e and "invalid_option" in e
            for e in errors
        ), f"Expected invalid strategy error, got: {errors}"

    def test_multi_chapter_assignment_strategy_valid_options(self):
        """Valid strategies: 'first', 'split', 'best_match'."""
        valid_strategies = ['first', 'split', 'best_match']
        for strategy in valid_strategies:
            config = self._make_config_with_chapter_grouping(
                multi_chapter_assignment_strategy=strategy
            )
            errors = config._validate_constraints()
            assert not any("multi_chapter_assignment_strategy" in e for e in errors), \
                f"Strategy '{strategy}' should be valid but got errors: {errors}"

    def test_chapter_grouping_disabled_with_matching_enabled_valid(self):
        """chapter_grouping disabled with matching enabled should not raise errors."""
        config = Config()
        config.matching.chapter_grouping.enabled = False
        config.matching.chapter_matching_enabled = True
        errors = config._validate_constraints()
        # Should not have any errors about chapter_grouping when disabled
        assert not any("chapter_grouping" in e for e in errors), \
            f"Disabled chapter_grouping should not cause errors: {errors}"


@pytest.mark.fast
class TestContextEnrichmentConstraints:
    """Test _validate_constraints covers context_enrichment fields."""

    def test_max_description_length_zero_returns_error(self):
        """AC3: validate() checks max_description_length is positive (0 fails)."""
        config = Config()
        config.matching.context_enrichment.max_description_length = 0
        errors = config._validate_constraints()
        assert any(
            "max_description_length" in e and "positive" in e
            for e in errors
        ), f"Expected max_description_length positive error, got: {errors}"

    def test_max_description_length_negative_returns_error(self):
        """Negative value caught by _validate_constraints."""
        config = Config()
        config.matching.context_enrichment.max_description_length = -5
        errors = config._validate_constraints()
        assert any("max_description_length" in e for e in errors)

    def test_max_description_length_positive_valid(self):
        """Positive value passes."""
        config = Config()
        # Default is 500, should pass
        errors = config._validate_constraints()
        assert not any("max_description_length" in e for e in errors)


@pytest.mark.fast
class TestCrossSectionConstraints:
    """Test cross-section constraint validation."""

    def test_title_enriched_embeddings_without_provider_returns_error(self):
        """AC4: title_enriched_embeddings=true requires embedding provider."""
        config = Config()
        config.matching.context_enrichment.title_enriched_embeddings = True
        config.embedding.provider = ""
        errors = config._validate_constraints()
        assert any(
            "title_enriched_embeddings" in e and "embedding.provider" in e
            for e in errors
        ), f"Expected cross-section error, got: {errors}"

    def test_title_enriched_embeddings_with_provider_valid(self):
        """title_enriched_embeddings=true with provider configured passes."""
        config = Config()
        config.matching.context_enrichment.title_enriched_embeddings = True
        config.embedding.provider = "gemini"
        errors = config._validate_constraints()
        assert not any("title_enriched_embeddings" in e for e in errors)

    def test_title_enriched_embeddings_false_no_provider_valid(self):
        """title_enriched_embeddings=false doesn't require provider."""
        config = Config()
        config.matching.context_enrichment.title_enriched_embeddings = False
        config.embedding.provider = ""
        errors = config._validate_constraints()
        assert not any("title_enriched_embeddings" in e for e in errors)
