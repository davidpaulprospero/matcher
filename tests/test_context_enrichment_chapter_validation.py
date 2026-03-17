"""Tests for ContextEnrichmentConfig and ChapterGroupingConfig __post_init__ validation (US-80-005)."""

import logging
import pytest

from src.config.sections.matching import ContextEnrichmentConfig, ChapterGroupingConfig


class TestContextEnrichmentConfigValidation:
    """Validate ContextEnrichmentConfig __post_init__ two-tier validation."""

    def test_negative_max_description_length_raises(self):
        """ValueError for impossible negative length."""
        with pytest.raises(ValueError, match="max_description_length=-1 is negative"):
            ContextEnrichmentConfig(max_description_length=-1)

    def test_zero_max_description_length_allowed(self):
        """Zero is valid (disables truncation effectively)."""
        cfg = ContextEnrichmentConfig(max_description_length=0)
        assert cfg.max_description_length == 0

    def test_normal_max_description_length_allowed(self):
        """Default value passes without issue."""
        cfg = ContextEnrichmentConfig()
        assert cfg.max_description_length == 500

    def test_exceeding_soft_limit_clamps_with_warning(self, caplog):
        """Values > 10000 are clamped with a warning."""
        with caplog.at_level(logging.WARNING):
            cfg = ContextEnrichmentConfig(max_description_length=20000)
        assert cfg.max_description_length == 10000
        assert "exceeds reasonable limit of 10000" in caplog.text

    def test_at_soft_limit_no_warning(self, caplog):
        """Exactly 10000 should not trigger warning."""
        with caplog.at_level(logging.WARNING):
            cfg = ContextEnrichmentConfig(max_description_length=10000)
        assert cfg.max_description_length == 10000
        assert "exceeds" not in caplog.text


class TestChapterGroupingConfigValidation:
    """Validate ChapterGroupingConfig __post_init__ two-tier validation."""

    def test_coherence_penalty_threshold_zero_raises(self):
        """ValueError when coherence_penalty_threshold < 1."""
        with pytest.raises(ValueError, match="coherence_penalty_threshold=0 must be >= 1"):
            ChapterGroupingConfig(coherence_penalty_threshold=0)

    def test_coherence_penalty_threshold_negative_raises(self):
        """ValueError when coherence_penalty_threshold is negative."""
        with pytest.raises(ValueError, match="must be >= 1"):
            ChapterGroupingConfig(coherence_penalty_threshold=-5)

    def test_coherence_penalty_threshold_one_allowed(self):
        """Minimum valid value of 1 is accepted."""
        cfg = ChapterGroupingConfig(coherence_penalty_threshold=1)
        assert cfg.coherence_penalty_threshold == 1

    def test_relevance_boost_weight_negative_raises(self):
        """ValueError for negative weight."""
        with pytest.raises(ValueError, match="relevance_boost_weight=-0.1 is negative"):
            ChapterGroupingConfig(relevance_boost_weight=-0.1)

    def test_relevance_boost_weight_exceeding_one_clamps_with_warning(self, caplog):
        """Values > 1.0 are clamped to 1.0 with warning."""
        with caplog.at_level(logging.WARNING):
            cfg = ChapterGroupingConfig(relevance_boost_weight=2.0)
        assert cfg.relevance_boost_weight == 1.0
        assert "relevance_boost_weight=2.0 exceeds 1.0" in caplog.text

    def test_relevance_boost_weight_at_one_no_warning(self, caplog):
        """Exactly 1.0 should not trigger warning."""
        with caplog.at_level(logging.WARNING):
            cfg = ChapterGroupingConfig(relevance_boost_weight=1.0)
        assert cfg.relevance_boost_weight == 1.0
        assert "exceeds" not in caplog.text

    def test_default_values_valid(self):
        """Default construction succeeds."""
        cfg = ChapterGroupingConfig()
        assert cfg.coherence_penalty_threshold == 5
        assert cfg.relevance_boost_weight == 0.1
