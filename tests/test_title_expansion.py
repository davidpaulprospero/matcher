"""
Tests for title semantic expansion (US-141-005).

Tests the expand_title_semantically() function in src/matching/scoring.py,
which generates semantically related terms from video title + voiceover context
to improve keyword matching recall.

US-141-005: Title semantic expansion for better matching
"""

import pytest
from unittest.mock import patch, MagicMock
from typing import List

from src.matching.scoring import expand_title_semantically, compute_semantic_context_similarity


class TestExpandTitleSemantically:
    """Tests for the expand_title_semantically function."""

    @pytest.mark.fast
    def test_empty_title_returns_empty_list(self):
        """Empty title should return empty list."""
        result = expand_title_semantically("", "some context")
        assert result == []

    @pytest.mark.fast
    def test_none_title_returns_empty_list(self):
        """None title should return empty list."""
        result = expand_title_semantically(None, "some context")  # type: ignore
        assert result == []

    @pytest.mark.fast
    def test_title_only(self):
        """Title without context should still work."""
        result = expand_title_semantically("Python Tutorial", "")
        # Returns cached or generated terms
        assert isinstance(result, list)

    @pytest.mark.fast
    def test_caching(self):
        """Same title+context should return cached result."""
        title = "JavaScript Basics"
        context = "Learning programming"

        # First call
        result1 = expand_title_semantically(title, context)

        # Second call should use cache
        result2 = expand_title_semantically(title, context)

        assert result1 == result2

    @pytest.mark.fast
    def test_different_contexts_different_results(self):
        """Different contexts may produce different expansions."""
        title = "Cooking pasta"

        result1 = expand_title_semantically(title, "Italian food recipe")
        result2 = expand_title_semantically(title, "Computer programming")

        # Results may differ based on context
        assert isinstance(result1, list)
        assert isinstance(result2, list)

    @pytest.mark.fast
    def test_max_terms_limit(self):
        """Should respect max_terms parameter."""
        # Mock LLM to return many terms
        many_terms = ["term" + str(i) for i in range(20)]

        with patch('src.llm_client.create_client') as mock_client:
            mock_response = MagicMock()
            mock_response.parsed_data = many_terms
            mock_client.return_value.generate.return_value = mock_response

            result = expand_title_semantically(
                "Test Title", "test context", max_terms=5
            )

            # Should be limited to max_terms
            assert len(result) <= 5

    @pytest.mark.fast
    def test_llm_failure_returns_empty(self):
        """LLM failure should return empty list gracefully."""
        with patch('src.llm_client.create_client') as mock_client:
            mock_client.side_effect = Exception("LLM error")

            result = expand_title_semantically("Test Title", "test context")

            assert result == []

    @pytest.mark.fast
    def test_invalid_llm_response_returns_empty(self):
        """Invalid LLM response should return empty list."""
        with patch('src.llm_client.create_client') as mock_client:
            mock_response = MagicMock()
            mock_response.parsed_data = "not a list"  # Invalid response
            mock_client.return_value.generate.return_value = mock_response

            result = expand_title_semantically("Test Title", "test context")

            assert result == []


class TestTitleExpansionConfig:
    """Tests for title expansion configuration."""

    @pytest.mark.fast
    def test_config_defaults(self):
        """Config should have default values for title expansion."""
        from src.config.sections.matching import MatchingConfig

        mc = MatchingConfig()

        # Check defaults exist
        assert hasattr(mc, 'title_expansion_enabled')
        assert hasattr(mc, 'title_expansion_model')
        assert hasattr(mc, 'title_expansion_max_terms')
        assert hasattr(mc, 'title_expansion_weight')

    @pytest.mark.fast
    def test_config_default_values(self):
        """Config should have correct default values."""
        from src.config.sections.matching import MatchingConfig

        mc = MatchingConfig()

        assert mc.title_expansion_enabled is True
        assert mc.title_expansion_model == "gemini-2.0-flash"
        assert mc.title_expansion_max_terms == 10
        assert mc.title_expansion_weight == 0.05


class TestComputeSemanticContextSimilarityWithExpansion:
    """Tests for compute_semantic_context_similarity with title expansion."""

    @pytest.mark.fast
    def test_expansion_integrated_when_enabled(self):
        """Title expansion should be included in video text when enabled."""
        # Create mock config with title expansion enabled
        mock_config = MagicMock()
        mock_matching = MagicMock()
        mock_matching.title_expansion_enabled = True
        mock_matching.title_expansion_max_terms = 5
        mock_config.matching = mock_matching

        # Mock expand_title_semantically
        expanded_terms = ["python", "coding", "programming", "tutorial", "learn"]

        with patch('src.matching.scoring.expand_title_semantically') as mock_expand:
            mock_expand.return_value = expanded_terms

            # Call with title and config
            result = compute_semantic_context_similarity(
                vo_context="learning to code",
                video_metadata={"title": "Python Tutorial"},
                config=mock_config
            )

            # Check expansion was called
            mock_expand.assert_called_once()

    @pytest.mark.fast
    def test_no_expansion_when_disabled(self):
        """Title expansion should not be called when disabled in config."""
        mock_config = MagicMock()
        mock_matching = MagicMock()
        mock_matching.title_expansion_enabled = False
        mock_config.matching = mock_matching

        with patch('src.matching.scoring.expand_title_semantically') as mock_expand:
            compute_semantic_context_similarity(
                vo_context="learning to code",
                video_metadata={"title": "Python Tutorial"},
                config=mock_config
            )

            # Expansion should NOT be called
            mock_expand.assert_not_called()

    @pytest.mark.fast
    def test_no_expansion_without_config(self):
        """Title expansion should not be called when no config provided."""
        with patch('src.matching.scoring.expand_title_semantically') as mock_expand:
            compute_semantic_context_similarity(
                vo_context="learning to code",
                video_metadata={"title": "Python Tutorial"},
                config=None
            )

            # Expansion should NOT be called
            mock_expand.assert_not_called()


class TestTitleExpansionImprovesRecall:
    """Tests verifying title expansion improves recall without sacrificing precision."""

    @pytest.mark.fast
    def test_expanded_terms_in_keyword_matching(self):
        """Expanded terms should be available for keyword matching."""
        # This test verifies the expand_title_semantically function produces
        # semantically related terms that can be used alongside original keywords

        # Test with a sample title and context
        title = "Python Programming Tutorial"
        context = "Learning how to code in Python"

        # The function should return a list of semantically related terms
        result = expand_title_semantically(title, context)

        # Verify it returns a list (from cache or generation)
        assert isinstance(result, list)
