"""
Tests for confidence-based gap categorization.

Tests for US-94-005: Add confidence-based gap categorization for smarter strategy selection
"""

import pytest
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.iterative_match.gap_analyzer import (
    GapSegment,
    GapAnalysis,
    ConfidenceCategory,
    categorize_gaps_by_confidence,
    get_strategy_for_confidence_category,
    DEFAULT_CONFIDENCE_THRESHOLDS,
    analyze_gaps,
)


# ============================================================================
# Test Fixtures
# ============================================================================

def create_gap_segment(
    segment_index: int,
    confidence: float,
    voiceover_text: str,
    position: float = 0.0,
) -> GapSegment:
    """Create a GapSegment for testing."""
    return GapSegment(
        segment_index=segment_index,
        confidence=confidence,
        voiceover_text=voiceover_text,
        position=position,
    )


# ============================================================================
# Test: categorize_gaps_by_confidence
# ============================================================================

class TestCategorizeGapsByConfidence:
    """Tests for categorize_gaps_by_confidence function."""

    def test_categorize_low_confidence_gaps(self):
        """Gaps with confidence < 0.3 should be categorized as low_confidence."""
        gaps = [
            create_gap_segment(0, 0.1, "Some text"),
            create_gap_segment(1, 0.25, "More text"),
            create_gap_segment(2, 0.05, "Another text"),
        ]

        result = categorize_gaps_by_confidence(gaps)

        assert len(result) == 3
        for gap in result:
            assert gap.confidence_category == "low_confidence"

    def test_categorize_medium_confidence_gaps(self):
        """Gaps with confidence 0.3-0.6 should be categorized as medium_confidence."""
        gaps = [
            create_gap_segment(0, 0.3, "Some text"),
            create_gap_segment(1, 0.45, "More text"),
            create_gap_segment(2, 0.59, "Another text"),
        ]

        result = categorize_gaps_by_confidence(gaps)

        assert len(result) == 3
        for gap in result:
            assert gap.confidence_category == "medium_confidence"

    def test_categorize_high_confidence_gaps(self):
        """Gaps with confidence > 0.6 should be categorized as high_confidence."""
        gaps = [
            create_gap_segment(0, 0.6, "Some text"),
            create_gap_segment(1, 0.75, "More text"),
            create_gap_segment(2, 0.95, "Another text"),
        ]

        result = categorize_gaps_by_confidence(gaps)

        assert len(result) == 3
        for gap in result:
            assert gap.confidence_category == "high_confidence"

    def test_categorize_mixed_confidence_gaps(self):
        """Gaps with mixed confidence levels should be categorized correctly."""
        gaps = [
            create_gap_segment(0, 0.1, "Low confidence text"),
            create_gap_segment(1, 0.45, "Medium confidence text"),
            create_gap_segment(2, 0.85, "High confidence text"),
        ]

        result = categorize_gaps_by_confidence(gaps)

        assert result[0].confidence_category == "low_confidence"
        assert result[1].confidence_category == "medium_confidence"
        assert result[2].confidence_category == "high_confidence"

    def test_categorize_with_custom_thresholds(self):
        """Custom thresholds should be used when provided."""
        gaps = [
            create_gap_segment(0, 0.2, "Text 1"),
            create_gap_segment(1, 0.5, "Text 2"),
            create_gap_segment(2, 0.8, "Text 3"),
        ]

        custom_thresholds = {"low": 0.2, "medium": 0.5, "high": 1.0}
        result = categorize_gaps_by_confidence(gaps, thresholds=custom_thresholds)

        # 0.2 should be at low boundary (not < 0.2, so medium)
        # 0.5 should be at medium boundary (not < 0.5, so high)
        # 0.8 should be high
        assert result[0].confidence_category == "medium_confidence"  # 0.2 >= 0.2
        assert result[1].confidence_category == "high_confidence"  # 0.5 >= 0.5
        assert result[2].confidence_category == "high_confidence"

    def test_categorize_empty_list(self):
        """Empty list should return empty list."""
        result = categorize_gaps_by_confidence([])
        assert result == []

    def test_default_thresholds(self):
        """Default thresholds should be 0.3 and 0.6."""
        assert DEFAULT_CONFIDENCE_THRESHOLDS["low"] == 0.3
        assert DEFAULT_CONFIDENCE_THRESHOLDS["medium"] == 0.6
        assert DEFAULT_CONFIDENCE_THRESHOLDS["high"] == 1.0


# ============================================================================
# Test: get_strategy_for_confidence_category
# ============================================================================

class TestGetStrategyForConfidenceCategory:
    """Tests for get_strategy_for_confidence_category function."""

    def test_low_confidence_strategy(self):
        """Low confidence gaps should get aggressive search strategy."""
        strategy = get_strategy_for_confidence_category("low_confidence")

        assert strategy["search_results"] == 15
        assert strategy["use_broad_queries"] is True
        assert strategy["max_iterations"] == 5
        assert strategy["parallel_strategies"] is True
        assert strategy["refine_on_fail"] is True

    def test_medium_confidence_strategy(self):
        """Medium confidence gaps should get standard search strategy."""
        strategy = get_strategy_for_confidence_category("medium_confidence")

        assert strategy["search_results"] == 10
        assert strategy["use_broad_queries"] is False
        assert strategy["max_iterations"] == 3
        assert strategy["parallel_strategies"] is True

    def test_high_confidence_strategy(self):
        """High confidence gaps should get minimal search strategy."""
        strategy = get_strategy_for_confidence_category("high_confidence")

        assert strategy["search_results"] == 5
        assert strategy["use_broad_queries"] is False
        assert strategy["max_iterations"] == 1
        assert strategy["parallel_strategies"] is False
        assert strategy["refine_on_fail"] is False

    def test_unknown_category_returns_medium(self):
        """Unknown category should return medium confidence strategy."""
        strategy = get_strategy_for_confidence_category("unknown")

        # Should default to medium
        assert strategy["search_results"] == 10


# ============================================================================
# Test: ConfidenceCategory Enum
# ============================================================================

class TestConfidenceCategory:
    """Tests for ConfidenceCategory enum."""

    def test_confidence_category_values(self):
        """ConfidenceCategory enum should have correct values."""
        assert ConfidenceCategory.LOW.value == "low_confidence"
        assert ConfidenceCategory.MEDIUM.value == "medium_confidence"
        assert ConfidenceCategory.HIGH.value == "high_confidence"


# ============================================================================
# Test: GapSegment with confidence_category
# ============================================================================

class TestGapSegmentConfidenceCategory:
    """Tests for GapSegment confidence_category field."""

    def test_gap_segment_default_confidence_category(self):
        """GapSegment should have empty confidence_category by default."""
        gap = create_gap_segment(0, 0.5, "Test text")
        assert gap.confidence_category == ""

    def test_gap_segment_confidence_category_after_categorization(self):
        """GapSegment confidence_category should be populated after categorization."""
        gap = create_gap_segment(0, 0.25, "Test text")
        gaps = categorize_gaps_by_confidence([gap])

        assert gaps[0].confidence_category == "low_confidence"


# ============================================================================
# Test: Integration with analyze_gaps
# ============================================================================

class TestConfidenceCategorizationIntegration:
    """Integration tests for confidence categorization with gap analysis."""

    def test_confidence_category_set_during_categorization(self):
        """Confidence category should be properly set for all gaps."""
        gaps = [
            create_gap_segment(0, 0.1, "freedom and justice"),
            create_gap_segment(1, 0.5, "John is running"),
            create_gap_segment(2, 0.8, "happy moment"),
        ]

        # Categorize by confidence
        categorized = categorize_gaps_by_confidence(gaps)

        assert categorized[0].confidence_category == "low_confidence"
        assert categorized[1].confidence_category == "medium_confidence"
        assert categorized[2].confidence_category == "high_confidence"

    def test_confidence_and_pattern_both_available(self):
        """Both pattern_type and confidence_category should be available."""
        gaps = [
            create_gap_segment(0, 0.15, "freedom and hope"),
        ]

        # First categorize by confidence
        categorized = categorize_gaps_by_confidence(gaps)

        # Then analyze patterns (mock entities)
        from unittest.mock import MagicMock
        mock_state = MagicMock()
        mock_state.entities = []

        analysis = analyze_gaps(categorized, state=mock_state)

        # Pattern should be set
        assert categorized[0].pattern_type == "abstract_concept"
        # And confidence category should be set
        assert categorized[0].confidence_category == "low_confidence"
