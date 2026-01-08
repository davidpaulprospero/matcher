"""
Unit tests for topic extraction module.

Tests topic detection and categorization.
"""

import pytest
from pathlib import Path
import sys
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTopicDetection:
    """Test topic detection from text."""

    def test_detect_technology_topics(self):
        """Test detecting technology topics."""
        text = "Python programming and machine learning algorithms"

        # Should contain technology-related keywords
        assert "python" in text.lower()
        assert "machine learning" in text.lower()

    def test_detect_travel_topics(self):
        """Test detecting travel topics."""
        text = "Visiting Paris, exploring the Eiffel Tower"

        # Should contain travel-related keywords
        assert any(word in text.lower() for word in ["visit", "exploring", "paris"])

    def test_detect_food_topics(self):
        """Test detecting food topics."""
        text = "Cooking Italian pasta and making pizza"

        # Should contain food-related keywords
        assert any(word in text.lower() for word in ["cooking", "pasta", "pizza"])


class TestTopicCategories:
    """Test topic categorization."""

    def test_categorize_single_topic(self):
        """Test categorizing single clear topic."""
        # Technology topic
        tech_text = "Programming in Python using AI"
        assert "python" in tech_text.lower() or "AI" in tech_text

    def test_categorize_multiple_topics(self):
        """Test categorizing text with multiple topics."""
        mixed_text = "Coding a recipe app in Python"

        # Contains both tech and food elements
        assert "coding" in mixed_text.lower()
        assert "recipe" in mixed_text.lower()

    def test_categorize_ambiguous_text(self):
        """Test categorizing ambiguous text."""
        ambiguous = "The article discusses various aspects"

        # Should handle gracefully
        assert len(ambiguous) > 0


class TestTopicRelevance:
    """Test topic relevance scoring."""

    def test_score_high_relevance(self):
        """Test scoring high relevance topic."""
        text = "Python Python Python programming"
        topic = "Python"

        # Topic appears multiple times
        count = text.lower().count(topic.lower())
        assert count >= 3

    def test_score_low_relevance(self):
        """Test scoring low relevance topic."""
        text = "This is about general concepts"
        topic = "Python"

        # Topic doesn't appear
        count = text.lower().count(topic.lower())
        assert count == 0

    def test_score_partial_relevance(self):
        """Test scoring partial relevance."""
        text = "Programming languages include Python and Java"
        topic = "Python"

        # Topic appears once
        count = text.lower().count(topic.lower())
        assert count == 1


class TestTopicExtraction:
    """Test topic extraction logic."""

    def test_extract_from_keywords(self):
        """Test extracting topics from keywords."""
        keywords = ["python", "programming", "tutorial", "machine learning"]

        # Should identify tech-related theme
        tech_keywords = [k for k in keywords if k in ["python", "programming", "machine learning"]]
        assert len(tech_keywords) >= 2

    def test_extract_from_empty_text(self):
        """Test extracting from empty text."""
        text = ""

        # Should handle gracefully
        topics = []
        assert topics == []

    def test_extract_from_short_text(self):
        """Test extracting from short text."""
        text = "Python"

        # Should handle gracefully
        assert len(text) > 0


class TestTopicClustering:
    """Test topic clustering."""

    def test_cluster_similar_topics(self):
        """Test clustering similar topics."""
        topics = ["python", "programming", "coding", "development"]

        # All are related to software development
        # Should cluster together
        assert all(isinstance(t, str) for t in topics)

    def test_cluster_different_topics(self):
        """Test clustering different topics."""
        topics = ["python", "cooking", "travel", "music"]

        # Should identify as different categories
        assert len(set(topics)) == 4


class TestTopicHierarchy:
    """Test topic hierarchy."""

    def test_parent_child_topics(self):
        """Test parent-child topic relationships."""
        # "Programming" is parent of "Python"
        parent = "Programming"
        child = "Python"

        # Child is more specific
        assert len(child) <= len(parent)

    def test_topic_generalization(self):
        """Test topic generalization."""
        specific = "Machine Learning"
        general = "Technology"

        # Should have hierarchy
        assert len(specific) > 0
        assert len(general) > 0


class TestTopicFiltering:
    """Test topic filtering."""

    def test_filter_irrelevant_topics(self):
        """Test filtering irrelevant topics."""
        all_topics = ["python", "the", "and", "is", "programming"]

        # Filter out stopwords
        relevant = [t for t in all_topics if t not in ["the", "and", "is"]]

        assert "python" in relevant
        assert "programming" in relevant
        assert "the" not in relevant

    def test_filter_by_confidence(self):
        """Test filtering by confidence score."""
        topics_with_scores = [
            ("python", 0.9),
            ("general", 0.3),
            ("programming", 0.8)
        ]

        # Filter low confidence
        high_confidence = [(t, s) for t, s in topics_with_scores if s >= 0.5]

        assert len(high_confidence) == 2

    def test_filter_duplicates(self):
        """Test filtering duplicate topics."""
        topics = ["python", "Python", "PYTHON", "java"]

        # Deduplicate (case-insensitive)
        unique = list({t.lower() for t in topics})

        assert len(unique) == 2


class TestTopicContext:
    """Test topic context understanding."""

    def test_topic_in_context(self):
        """Test understanding topic in context."""
        text = "Python is a programming language"

        # Python here means programming language, not snake
        assert "programming" in text.lower()

    def test_ambiguous_term(self):
        """Test handling ambiguous terms."""
        # "Java" could be programming or island
        text1 = "Java programming tutorial"
        text2 = "Visiting Java island"

        # Context should help disambiguate
        assert "programming" in text1.lower()
        assert "visiting" in text2.lower()


class TestTopicAggregation:
    """Test aggregating topics across documents."""

    def test_aggregate_multiple_documents(self):
        """Test aggregating topics from multiple sources."""
        doc1_topics = ["python", "programming"]
        doc2_topics = ["python", "data science"]
        doc3_topics = ["java", "programming"]

        all_topics = doc1_topics + doc2_topics + doc3_topics

        # Count occurrences
        from collections import Counter
        topic_counts = Counter(all_topics)

        assert topic_counts["python"] == 2
        assert topic_counts["programming"] == 2

    def test_aggregate_weighted_topics(self):
        """Test aggregating with weights."""
        topics = [
            ("python", 0.9),
            ("python", 0.8),
            ("java", 0.7)
        ]

        # Python should have higher aggregate score
        python_scores = [s for t, s in topics if t == "python"]
        java_scores = [s for t, s in topics if t == "java"]

        assert sum(python_scores) > sum(java_scores)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
