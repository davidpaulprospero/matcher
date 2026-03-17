"""
Tests for US-157-004: Search Query Optimization

Tests cover:
- Query preprocessing (stopwords, whitespace, special characters)
- Query expansion with related terms
- Relevance scoring for search results
- max_query_length truncation
"""

import pytest
from unittest.mock import Mock
from src.stages.video_search import VideoSearchStage
from src.config.sections.video_search import VideoSearchConfig


class TestQueryPreprocessing:
    """Tests for query preprocessing functionality."""

    @pytest.fixture
    def stage(self):
        """Create VideoSearchStage instance for testing."""
        return VideoSearchStage()

    @pytest.fixture
    def config_with_stopwords(self):
        """Config with stopword removal enabled."""
        return VideoSearchConfig(
            enable_stopword_removal=True,
            stopword_list=['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at']
        )

    @pytest.fixture
    def config_without_stopwords(self):
        """Config with stopword removal disabled."""
        return VideoSearchConfig(enable_stopword_removal=False)

    def test_remove_stopwords(self, stage, config_with_stopwords):
        """Test that stopwords are removed from query."""
        query = "the beach sunset at the ocean"
        result = stage._preprocess_query(query, config_with_stopwords)
        assert result == "beach sunset ocean"

    def test_preserve_content_without_stopwords(self, stage, config_without_stopwords):
        """Test that content is preserved when stopword removal disabled."""
        query = "the beach sunset at the ocean"
        result = stage._preprocess_query(query, config_without_stopwords)
        assert "the" in result
        assert "beach" in result

    def test_normalize_whitespace(self, stage, config_with_stopwords):
        """Test that multiple whitespaces are normalized."""
        query = "beach    sunset   waves"
        result = stage._preprocess_query(query, config_with_stopwords)
        assert "  " not in result
        assert result == "beach sunset waves"

    def test_handle_special_characters(self, stage, config_with_stopwords):
        """Test that special characters are handled."""
        query = "beach@sunset! #ocean #waves"
        result = stage._preprocess_query(query, config_with_stopwords)
        # Special chars should be removed
        assert "@" not in result
        assert "!" not in result
        assert "#" not in result

    def test_empty_query(self, stage, config_with_stopwords):
        """Test empty query returns empty."""
        result = stage._preprocess_query("", config_with_stopwords)
        assert result == ""

    def test_none_query(self, stage, config_with_stopwords):
        """Test None query returns None."""
        result = stage._preprocess_query(None, config_with_stopwords)
        assert result is None


class TestQueryExpansion:
    """Tests for query expansion functionality."""

    @pytest.fixture
    def stage(self):
        """Create VideoSearchStage instance for testing."""
        return VideoSearchStage()

    @pytest.fixture
    def config_with_topic_tags(self):
        """Config with topic tags for expansion."""
        return VideoSearchConfig(
            enable_query_expansion=True,
            topic_tags={
                'nature': ['wildlife', 'landscape', 'outdoor'],
                'travel': ['adventure', 'destination', 'culture'],
                'beach': ['ocean', 'waves', 'sand'],
            }
        )

    @pytest.fixture
    def config_expansion_disabled(self):
        """Config with query expansion disabled."""
        return VideoSearchConfig(enable_query_expansion=False)

    def test_expand_query_with_matching_topic(self, stage, config_with_topic_tags):
        """Test query expansion with matching topic."""
        query = "beach sunset"
        result = stage._expand_query(query, config_with_topic_tags)
        # Should expand with related terms from beach
        assert "ocean" in result or "waves" in result or "sand" in result

    def test_no_expansion_when_disabled(self, stage, config_expansion_disabled):
        """Test no expansion when disabled."""
        query = "beach sunset"
        result = stage._expand_query(query, config_expansion_disabled)
        assert result == query

    def test_expand_with_topic_parameter(self, stage, config_with_topic_tags):
        """Test query expansion with topic parameter."""
        query = "vacation"
        result = stage._expand_query(query, config_with_topic_tags, topic='travel')
        # Should expand with travel-related terms
        assert "adventure" in result or "destination" in result or "culture" in result


class TestQueryLengthTruncation:
    """Tests for max_query_length truncation."""

    @pytest.fixture
    def stage(self):
        """Create VideoSearchStage instance for testing."""
        return VideoSearchStage()

    @pytest.fixture
    def config_256_chars(self):
        """Config with 256 char max."""
        return VideoSearchConfig(max_query_length=256)

    @pytest.fixture
    def config_50_chars(self):
        """Config with 50 char max for testing."""
        return VideoSearchConfig(max_query_length=50)

    def test_truncate_long_query(self, stage, config_50_chars):
        """Test truncation of long query."""
        query = "this is a very long query that definitely exceeds fifty characters and should be truncated"
        result = stage._truncate_query_length(query, config_50_chars)
        assert len(result) <= 50

    def test_preserve_short_query(self, stage, config_256_chars):
        """Test short query is preserved."""
        query = "beach sunset"
        result = stage._truncate_query_length(query, config_256_chars)
        assert result == query


class TestRelevanceScoring:
    """Tests for relevance scoring functionality."""

    @pytest.fixture
    def stage(self):
        """Create VideoSearchStage instance for testing."""
        return VideoSearchStage()

    def test_high_relevance_exact_match(self, stage):
        """Test high relevance for exact title match."""
        title = "beach sunset ocean waves"
        description = "relaxing beach scene"
        query = "beach sunset"

        score = stage._calculate_relevance_score(title, description, query)
        assert score > 0.7

    def test_low_relevance_no_match(self, stage):
        """Test low relevance for no match."""
        title = "cooking tutorial"
        description = "how to make pasta"
        query = "beach vacation"

        score = stage._calculate_relevance_score(title, description, query)
        assert score < 0.3

    def test_partial_match(self, stage):
        """Test relevance for partial match."""
        title = "beach nature timelapse"
        description = "beautiful nature scenery"
        query = "nature"

        score = stage._calculate_relevance_score(title, description, query)
        assert score > 0.3

    def test_empty_title(self, stage):
        """Test zero score for empty title."""
        score = stage._calculate_relevance_score("", "description", "query")
        assert score == 0.0

    def test_empty_query(self, stage):
        """Test zero score for empty query."""
        score = stage._calculate_relevance_score("title", "description", "")
        assert score == 0.0


class TestRelevanceFiltering:
    """Tests for relevance filtering of results."""

    @pytest.fixture
    def stage(self):
        """Create VideoSearchStage instance for testing."""
        return VideoSearchStage()

    @pytest.fixture
    def config_with_min_score(self):
        """Config with minimum relevance score."""
        return VideoSearchConfig(
            enable_relevance_scoring=True,
            min_relevance_score=0.3
        )

    @pytest.fixture
    def sample_results(self):
        """Sample search results."""
        return [
            {'video_id': '1', 'title': 'beach sunset', 'description': 'ocean waves'},
            {'video_id': '2', 'title': 'cooking pasta', 'description': 'recipe tutorial'},
            {'video_id': '3', 'title': 'beach vacation', 'description': 'travel guide'},
        ]

    def test_filter_low_relevance(self, stage, config_with_min_score):
        """Test filtering of low relevance results."""
        results = [
            {'video_id': '1', 'title': 'beach sunset ocean', 'description': 'relaxing'},
            {'video_id': '2', 'title': 'completely unrelated video', 'description': 'random'},
        ]
        query = "beach sunset"

        filtered = stage._score_results_by_relevance(results, query, config_with_min_score)
        # High relevance should pass, low should be filtered
        assert len(filtered) >= 1

    def test_scoring_disabled_returns_all(self, stage, sample_results):
        """Test that scoring disabled returns all results."""
        config = Mock()
        config.enable_relevance_scoring = False

        filtered = stage._score_results_by_relevance(sample_results, "test", config)
        assert len(filtered) == len(sample_results)

    def test_relevance_score_added_to_results(self, stage, config_with_min_score):
        """Test that relevance score is added to results."""
        results = [
            {'video_id': '1', 'title': 'beach sunset', 'description': 'ocean'},
        ]
        query = "beach sunset"

        scored = stage._score_results_by_relevance(results, query, config_with_min_score)
        assert 'relevance_score' in scored[0]


class TestQueryOptimizationIntegration:
    """Integration tests for full query optimization pipeline."""

    @pytest.fixture
    def stage(self):
        """Create VideoSearchStage instance for testing."""
        return VideoSearchStage()

    @pytest.fixture
    def full_config(self):
        """Full config with all optimization enabled."""
        return VideoSearchConfig(
            max_query_length=256,
            enable_stopword_removal=True,
            enable_query_expansion=True,
            enable_relevance_scoring=True,
            min_relevance_score=0.3,
            topic_tags={
                'nature': ['wildlife', 'landscape'],
                'travel': ['adventure', 'destination'],
            }
        )

    def test_full_optimization_pipeline(self, stage, full_config):
        """Test full query optimization pipeline."""
        # Start with raw query
        raw_query = "the best @nature #travel video!"

        # Apply preprocessing
        preprocessed = stage._preprocess_query(raw_query, full_config)
        assert "@" not in preprocessed
        assert "#" not in preprocessed

        # Apply expansion
        expanded = stage._expand_query(preprocessed, full_config)
        # Should add related terms

        # Apply truncation
        truncated = stage._truncate_query_length(expanded, full_config)
        assert len(truncated) <= 256

    def test_optimization_respects_config(self, stage):
        """Test that optimization respects config settings."""
        # Config with expansion disabled
        config = Mock()
        config.enable_stopword_removal = False
        config.enable_query_expansion = False
        config.max_query_length = 100
        config.enable_relevance_scoring = True
        config.min_relevance_score = 0.3
        config.relevance_boost_factor = 0.1

        query = "test query"
        result = stage._preprocess_query(query, config)
        assert result == query  # No stopword removal
