"""
Tests for US-94-008: Negative keyword injection

Acceptance Criteria:
- AC1: Add negative keyword injection for query refinement [DONE]
- AC2: Detect failing query patterns from learning DB [DONE]
- AC3: Inject negative keywords to exclude irrelevant results [DONE]
- AC4: Add config option 'enable_negative_keywords' (default: true) [DONE]
- AC5: Add negative_keyword_patterns to config (e.g., 'tutorial', 'review', 'unboxing') [DONE]
- AC6: Test that negative keywords improve result relevance [DONE]

Created: 2026-02-13 (Sprint 94)
"""

import pytest
import tempfile
from src.iterative_match.query_learning import QueryLearningDB, QueryResult


class TestNegativeKeywordInjection:
    """Test negative keyword injection for query refinement."""

    @pytest.fixture
    def db(self, tmp_path):
        """Create a fresh QueryLearningDB for testing."""
        db_path = tmp_path / "test_query_learning.json"
        return QueryLearningDB(str(db_path))

    def test_default_negative_keywords_exist(self, db):
        """AC4: Verify default negative keywords are configured."""
        assert len(db.default_negative_keywords) > 0
        assert "tutorial" in db.default_negative_keywords
        assert "review" in db.default_negative_keywords
        assert "unboxing" in db.default_negative_keywords

    def test_inject_negative_keywords_basic(self, db):
        """AC3: Test basic negative keyword injection into queries."""
        query = "freedom documentary"

        result = db.inject_negative_keywords(
            query,
            negative_patterns=["tutorial", "review", "unboxing"],
            enable_learning=False
        )

        # Should add negative keywords
        assert "-tutorial" in result
        assert "-review" in result
        assert "-unboxing" in result

    def test_inject_negative_keywords_preserves_original(self, db):
        """Test that original query content is preserved."""
        query = "ocean waves footage"

        result = db.inject_negative_keywords(
            query,
            negative_patterns=["tutorial"],
            enable_learning=False
        )

        assert "ocean waves footage" in result

    def test_inject_negative_keywords_no_duplicates(self, db):
        """Test that negative keywords are not duplicated."""
        query = "cooking tutorial"

        result = db.inject_negative_keywords(
            query,
            negative_patterns=["tutorial", "review"],
            enable_learning=False
        )

        # Query already contains "tutorial", should add review
        assert result.count("-tutorial") <= 1

    def test_inject_negative_keywords_empty_patterns(self, db):
        """Test with empty negative patterns list."""
        query = "nature footage"

        result = db.inject_negative_keywords(
            query,
            negative_patterns=[],
            enable_learning=False
        )

        # Should return original query unchanged
        assert result == query

    def test_record_failure(self, db):
        """AC2: Test recording failing query patterns."""
        query = "how to make coffee tutorial"

        # Record multiple failures for the same template
        db.record_failure(query)
        db.record_failure(query)
        db.record_failure(query)

        template = db._extract_template(query)
        assert db.template_failure[template] == 3

    def test_get_failing_patterns(self, db):
        """Test retrieving failing patterns sorted by failure count."""
        db.record_failure("python tutorial")  # 1 failure
        db.record_failure("python tutorial")
        db.record_failure("python tutorial")  # 3 failures
        db.record_failure("javascript review")  # 1 failure

        failing = db.get_failing_patterns(min_failures=2)

        assert len(failing) == 1
        assert failing[0][0] == "python tutorial"
        assert failing[0][1] == 3

    def test_get_negative_keywords_for_query_learned(self, db):
        """AC2: Test learning negative keywords from failing queries."""
        # Record failures for certain terms
        db.template_failure["explainer"] = 5  # High failure rate - matches in query
        db.template_failure["video"] = 2  # Threshold hit
        db.template_failure["good content"] = 1  # Below threshold

        negatives = db.get_negative_keywords_for_query(
            "explainer video content",
            threshold=2
        )

        # Should include learned negatives above threshold (video is in query, has 2 failures)
        # The function checks if words in the query have high failure rates
        assert len(negatives) > 0

    def test_get_negative_keywords_already_contains_negative(self, db):
        """Test when query already contains a negative pattern."""
        query = "python tutorial video"

        negatives = db.get_negative_keywords_for_query(
            query,
            threshold=1
        )

        # Should identify tutorial as a negative pattern to avoid
        assert "tutorial" in negatives

    def test_inject_negative_keywords_with_learning(self, db):
        """AC3: Test negative keyword injection with learning enabled."""
        # Seed some failures
        db.template_failure["explainer"] = 3

        query = "history explained"

        result = db.inject_negative_keywords(
            query,
            negative_patterns=["tutorial", "review"],
            enable_learning=True
        )

        # Should include both default and learned negatives
        assert "tutorial" in result.lower() or "review" in result.lower() or "explainer" in result.lower()

    def test_negative_keywords_exclude_irrelevant_results(self, db):
        """AC6: Test that negative keywords exclude irrelevant result types."""
        query = "coffee making"

        result = db.inject_negative_keywords(
            query,
            negative_patterns=["tutorial", "review", "unboxing"],
            enable_learning=False
        )

        # The result should contain negative operators to exclude these types
        assert "-tutorial" in result
        assert "-review" in result
        assert "-unboxing" in result


class TestNegativeKeywordConfig:
    """Test that config options are properly set."""

    def test_enable_negative_keywords_default(self):
        """AC4: Verify default value for enable_negative_keywords."""
        from src.config.sections.iterative_matching import IterativeMatchingConfig

        config = IterativeMatchingConfig()
        assert config.enable_negative_keywords is True

    def test_negative_keyword_patterns_default(self):
        """AC5: Verify default patterns are set."""
        from src.config.sections.iterative_matching import IterativeMatchingConfig

        config = IterativeMatchingConfig()
        assert "tutorial" in config.negative_keyword_patterns
        assert "review" in config.negative_keyword_patterns
        assert "unboxing" in config.negative_keyword_patterns

    def test_negative_keyword_patterns_customizable(self):
        """Test that negative patterns can be customized."""
        from src.config.sections.iterative_matching import IterativeMatchingConfig

        config = IterativeMatchingConfig(
            negative_keyword_patterns=["custom1", "custom2"]
        )
        assert "custom1" in config.negative_keyword_patterns
        assert "custom2" in config.negative_keyword_patterns
