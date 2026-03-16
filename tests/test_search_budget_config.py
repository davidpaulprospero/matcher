"""Tests for SearchBudgetConfig and budget calculation logic."""

import pytest
from src.config.sections.video_search import SearchBudgetConfig, VideoSearchConfig


class TestSearchBudgetConfig:
    """Tests for SearchBudgetConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SearchBudgetConfig()
        assert config.max_total_results == 200
        assert config.results_per_keyword == 20

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SearchBudgetConfig(
            max_total_results=100,
            results_per_keyword=10
        )
        assert config.max_total_results == 100
        assert config.results_per_keyword == 10


class TestBudgetCalculation:
    """Tests for budget distribution calculation.

    This mirrors the logic in Get-DistributedKeywordBudget in interview.ps1
    to ensure Python and PowerShell calculations are consistent.
    """

    @staticmethod
    def calculate_distributed_budget(
        keywords: list,
        max_total_results: int = 200,
        results_per_keyword: int = 20
    ) -> dict:
        """Calculate adjusted results per keyword based on budget.

        This is a Python port of Get-DistributedKeywordBudget from interview.ps1.

        Args:
            keywords: List of keyword strings
            max_total_results: Maximum total results from config
            results_per_keyword: Desired results per keyword from config

        Returns:
            Dictionary with adjustedResultsPerKeyword, totalKeywords,
            willReduce, warningMessage, effectiveTotal
        """
        if len(keywords) == 0:
            return {
                'adjustedResultsPerKeyword': results_per_keyword,
                'totalKeywords': 0,
                'willReduce': False,
                'warningMessage': 'No keywords provided',
                'effectiveTotal': 0
            }

        total_keywords = len(keywords)
        budget_threshold = max_total_results // results_per_keyword

        if total_keywords <= budget_threshold:
            # Keywords fit within budget
            return {
                'adjustedResultsPerKeyword': results_per_keyword,
                'totalKeywords': total_keywords,
                'willReduce': False,
                'warningMessage': None,
                'effectiveTotal': total_keywords * results_per_keyword
            }

        # Keywords exceed budget - need to reduce
        adjusted_results = max_total_results // total_keywords
        effective_total = total_keywords * adjusted_results

        warning_msg = (
            f"WARNING: {total_keywords} keywords exceeds budget of {budget_threshold}. "
            f"Results per keyword will be reduced from {results_per_keyword} to {adjusted_results} "
            f"to stay within {max_total_results} limit."
        )

        return {
            'adjustedResultsPerKeyword': adjusted_results,
            'totalKeywords': total_keywords,
            'willReduce': True,
            'warningMessage': warning_msg,
            'effectiveTotal': effective_total
        }

    def test_5_keywords_within_budget(self):
        """Test: 5 keywords -> 20 results each (within budget of 200).

        5 * 20 = 100 <= 200, so no adjustment needed.
        """
        keywords = [f"keyword{i}" for i in range(5)]
        result = self.calculate_distributed_budget(
            keywords,
            max_total_results=200,
            results_per_keyword=20
        )

        assert result['adjustedResultsPerKeyword'] == 20
        assert result['totalKeywords'] == 5
        assert result['willReduce'] is False
        assert result['warningMessage'] is None
        assert result['effectiveTotal'] == 100

    def test_15_keywords_requires_adjustment(self):
        """Test: 15 keywords -> 13 results each (exceeds budget).

        15 * 20 = 300 > 200, so needs adjustment: 200 // 15 = 13
        """
        keywords = [f"keyword{i}" for i in range(15)]
        result = self.calculate_distributed_budget(
            keywords,
            max_total_results=200,
            results_per_keyword=20
        )

        assert result['adjustedResultsPerKeyword'] == 13
        assert result['totalKeywords'] == 15
        assert result['willReduce'] is True
        assert result['warningMessage'] is not None
        assert result['effectiveTotal'] == 195  # 15 * 13

    def test_10_keywords_at_threshold(self):
        """Test: 10 keywords -> 20 results each (exactly at threshold).

        10 * 20 = 200, exactly at budget limit.
        """
        keywords = [f"keyword{i}" for i in range(10)]
        result = self.calculate_distributed_budget(
            keywords,
            max_total_results=200,
            results_per_keyword=20
        )

        assert result['adjustedResultsPerKeyword'] == 20
        assert result['totalKeywords'] == 10
        assert result['willReduce'] is False
        assert result['effectiveTotal'] == 200

    def test_empty_keywords(self):
        """Test: 0 keywords returns defaults."""
        result = self.calculate_distributed_budget(
            [],
            max_total_results=200,
            results_per_keyword=20
        )

        assert result['adjustedResultsPerKeyword'] == 20
        assert result['totalKeywords'] == 0
        assert result['willReduce'] is False

    def test_single_keyword(self):
        """Test: 1 keyword gets full budget."""
        keywords = ["single_keyword"]
        result = self.calculate_distributed_budget(
            keywords,
            max_total_results=200,
            results_per_keyword=20
        )

        assert result['adjustedResultsPerKeyword'] == 20
        assert result['totalKeywords'] == 1
        assert result['willReduce'] is False

    def test_custom_max_total_results(self):
        """Test with different max_total_results."""
        keywords = [f"keyword{i}" for i in range(20)]

        # With max_total_results=100 and 20 keywords
        # 100 // 20 = 5 results per keyword
        result = self.calculate_distributed_budget(
            keywords,
            max_total_results=100,
            results_per_keyword=20
        )

        assert result['adjustedResultsPerKeyword'] == 5
        assert result['totalKeywords'] == 20
        assert result['willReduce'] is True
        assert result['effectiveTotal'] == 100


class TestVideoSearchConfig:
    """Tests for VideoSearchConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = VideoSearchConfig()
        assert config.results_per_keyword == 20
        assert config.max_total_results == 200
        assert config.search_budget_aware is True
        assert config.auto_distribute_budget is True

    def test_nested_defaults(self):
        """Test that nested fields get proper defaults."""
        config = VideoSearchConfig()
        assert config.negative_keywords is not None
        assert "trailer" in config.negative_keywords
        assert config.topic_tags is not None
        assert "nature" in config.topic_tags

    def test_custom_values(self):
        """Test custom configuration values."""
        config = VideoSearchConfig(
            results_per_keyword=50,
            max_total_results=500,
            search_budget_aware=False
        )
        assert config.results_per_keyword == 50
        assert config.max_total_results == 500
        assert config.search_budget_aware is False
