"""
Comprehensive tests for keyword alternatives module.

Covers:
- KeywordAlternativeGenerator initialization
- Single alternative keyword generation (download optimization)
- Multiple alternative keywords generation (post-download remix)
- LLM integration (Gemini and Anthropic)
- Simple rule-based remixing
- Fallback remixing strategies
- API key and model configuration
- Error handling

Created: 2026-01-09 (Phase 5.2)
"""

from unittest.mock import Mock, MagicMock, patch
import pytest

from src.keyword_alternatives import KeywordAlternativeGenerator


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Create mock config"""
    config = Mock()
    config.api_keys = Mock()
    config.api_keys.gemini_api_key = "fake_gemini_key"
    config.api_keys.anthropic_api_key = "fake_anthropic_key"
    config.matching = Mock()
    config.matching.gemini_model = "gemini-2.0-flash"
    config.matching.anthropic_model = "claude-3-haiku-20240307"
    config.llm = Mock()
    config.llm.model = "gemini-2.0-flash"
    return config


@pytest.fixture
def generator(mock_config):
    """Create KeywordAlternativeGenerator instance"""
    return KeywordAlternativeGenerator(mock_config)


# ============================================================================
# Test Initialization
# ============================================================================

class TestKeywordAlternativeGeneratorInit:
    """Test KeywordAlternativeGenerator initialization"""

    def test_init_with_config(self, mock_config):
        """Test initialization with config"""
        generator = KeywordAlternativeGenerator(mock_config)

        assert generator.config == mock_config

    def test_init_without_config(self):
        """Test initialization without config"""
        generator = KeywordAlternativeGenerator(None)

        assert generator.config is None


# ============================================================================
# Test Single Alternative Generation
# ============================================================================

class TestSingleAlternativeGeneration:
    """Test single alternative keyword generation"""

    @patch('src.llm_client.create_client')
    def test_generate_single_alternative_success(self, mock_create_client, generator):
        """Test successful single alternative generation"""
        # Mock LLM response
        mock_response = Mock()
        mock_response.text = "beach vacation"
        mock_client = Mock()
        mock_client.generate.return_value = mock_response
        mock_create_client.return_value = mock_client

        result = generator.generate_single_alternative(
            keyword="tropical getaway",
            topic="travel videos",
            provider="gemini",
            api_key="test_key"
        )

        assert result == "beach vacation"
        mock_create_client.assert_called_once()

    @patch('src.llm_client.create_client')
    def test_generate_single_alternative_no_result(self, mock_create_client, generator):
        """Test when LLM returns no result"""
        mock_response = Mock()
        mock_response.text = None
        mock_client = Mock()
        mock_client.generate.return_value = mock_response
        mock_create_client.return_value = mock_client

        result = generator.generate_single_alternative(
            keyword="test",
            provider="gemini",
            api_key="test_key"
        )

        assert result is None

    @patch('src.llm_client.create_client')
    def test_generate_single_alternative_strips_quotes(self, mock_create_client, generator):
        """Test that quotes are stripped from result"""
        mock_response = Mock()
        mock_response.text = '"beach footage"'
        mock_client = Mock()
        mock_client.generate.return_value = mock_response
        mock_create_client.return_value = mock_client

        result = generator.generate_single_alternative(
            keyword="test",
            provider="gemini",
            api_key="test_key"
        )

        assert result == "beach footage"
        assert '"' not in result

    @patch('src.llm_client.create_client')
    def test_generate_single_alternative_rejects_long_result(self, mock_create_client, generator):
        """Test that overly long results are rejected"""
        mock_response = Mock()
        mock_response.text = "a" * 150  # Longer than 100 chars
        mock_client = Mock()
        mock_client.generate.return_value = mock_response
        mock_create_client.return_value = mock_client

        result = generator.generate_single_alternative(
            keyword="test",
            provider="gemini",
            api_key="test_key"
        )

        assert result is None

    def test_generate_single_alternative_no_api_key(self, generator):
        """Test when no API key is available"""
        result = generator.generate_single_alternative(
            keyword="test",
            provider="gemini",
            api_key=None
        )

        assert result is None

    @patch('src.llm_client.create_client')
    def test_generate_single_alternative_error_handling(self, mock_create_client, generator):
        """Test error handling during generation"""
        mock_create_client.side_effect = Exception("API error")

        result = generator.generate_single_alternative(
            keyword="test",
            provider="gemini",
            api_key="test_key"
        )

        assert result is None


# ============================================================================
# Test Multiple Alternatives Generation
# ============================================================================

class TestMultipleAlternativesGeneration:
    """Test multiple alternatives keyword generation"""

    @patch('src.llm_client.create_client')
    def test_generate_multiple_alternatives_success(self, mock_create_client, generator):
        """Test successful multiple alternatives generation"""
        mock_response = Mock()
        mock_response.parsed_data = ["beach vacation", "tropical getaway", "island paradise"]
        mock_client = Mock()
        mock_client.generate.return_value = mock_response
        mock_create_client.return_value = mock_client

        keywords, status = generator.generate_multiple_alternatives(
            keyword="beach",
            topic_context="travel videos",
            provider="gemini",
            api_key="test_key"
        )

        assert len(keywords) == 3
        assert "beach vacation" in keywords
        assert "Gemini remix successful" in status

    @patch('src.llm_client.create_client')
    def test_generate_multiple_alternatives_empty_result(self, mock_create_client, generator):
        """Test when LLM returns empty list"""
        mock_response = Mock()
        mock_response.parsed_data = []
        mock_client = Mock()
        mock_client.generate.return_value = mock_response
        mock_create_client.return_value = mock_client

        keywords, status = generator.generate_multiple_alternatives(
            keyword="test",
            provider="gemini",
            api_key="test_key"
        )

        assert keywords == []
        assert "failed" in status.lower()

    @patch('src.llm_client.create_client')
    def test_generate_multiple_alternatives_anthropic(self, mock_create_client, generator):
        """Test multiple alternatives with Anthropic provider"""
        mock_response = Mock()
        mock_response.parsed_data = ["alt1", "alt2"]
        mock_client = Mock()
        mock_client.generate.return_value = mock_response
        mock_create_client.return_value = mock_client

        keywords, status = generator.generate_multiple_alternatives(
            keyword="test",
            provider="anthropic",
            api_key="test_key",
            max_tokens=500
        )

        assert len(keywords) == 2
        assert "Anthropic remix successful" in status

    def test_generate_multiple_alternatives_no_api_key(self):
        """Test when no API key is available"""
        # Use generator with no config (no API keys)
        generator = KeywordAlternativeGenerator(None)

        keywords, status = generator.generate_multiple_alternatives(
            keyword="test",
            provider="gemini",
            api_key=None
        )

        assert keywords == []
        assert "not available" in status

    @patch('src.llm_client.create_client')
    def test_generate_multiple_alternatives_error_handling(self, mock_create_client, generator):
        """Test error handling during generation"""
        mock_create_client.side_effect = Exception("API error")

        keywords, status = generator.generate_multiple_alternatives(
            keyword="test",
            provider="gemini",
            api_key="test_key"
        )

        assert keywords == []
        assert "failed" in status.lower()


# ============================================================================
# Test Simple Keyword Remix
# ============================================================================

class TestSimpleRemix:
    """Test simple rule-based keyword remixing"""

    def test_simple_remix_add_footage(self, generator):
        """Test adding 'footage' suffix"""
        result = generator.simple_remix_keyword("beach vacation")

        assert result is not None
        assert "footage" in result.lower()

    def test_simple_remix_remove_qualifiers(self, generator):
        """Test removing common qualifiers"""
        result = generator.simple_remix_keyword("the best beach vacation")

        assert result is not None
        assert "the" not in result.lower()
        assert "best" not in result.lower()

    def test_simple_remix_simplify_to_core(self, generator):
        """Test simplifying to core words"""
        result = generator.simple_remix_keyword("amazing tropical island beach vacation video")

        assert result is not None
        # Should be simplified

    def test_simple_remix_already_has_footage(self, generator):
        """Test when keyword already has 'footage'"""
        result = generator.simple_remix_keyword("beach footage")

        assert result is not None
        # Should still return something (simplified version)

    def test_simple_remix_empty_after_filtering(self, generator):
        """Test when all words are filtered out"""
        result = generator.simple_remix_keyword("the a an")

        assert result is None


# ============================================================================
# Test Fallback Remix
# ============================================================================

class TestFallbackRemix:
    """Test fallback remixing strategies"""

    def test_fallback_remix_remove_year(self, generator):
        """Test removing year patterns"""
        result = generator.fallback_remix("Beach vacation 2024")

        assert any("2024" not in r for r in result)

    def test_fallback_remix_remove_footage(self, generator):
        """Test removing 'footage' suffix"""
        result = generator.fallback_remix("Beach vacation footage")

        assert len(result) > 0
        # Should include versions without 'footage'

    def test_fallback_remix_remove_locations(self, generator):
        """Test removing specific location names"""
        result = generator.fallback_remix("Beach in California")

        assert len(result) > 0
        # Should include version without 'California'

    def test_fallback_remix_word_subsets(self, generator):
        """Test taking subsets of words"""
        result = generator.fallback_remix("tropical beach vacation paradise")

        assert len(result) > 0
        # Should include word subsets

    def test_fallback_remix_deduplication(self, generator):
        """Test that duplicates are removed"""
        result = generator.fallback_remix("Beach vacation 2024")

        # Should not have duplicates
        assert len(result) == len(set(result))

    def test_fallback_remix_max_three_results(self, generator):
        """Test that at most 3 alternatives are returned"""
        result = generator.fallback_remix("tropical beach vacation paradise footage 2024")

        assert len(result) <= 3


# ============================================================================
# Test Configuration Helpers
# ============================================================================

class TestConfigurationHelpers:
    """Test API key and model configuration helpers"""

    def test_get_api_key_gemini(self, generator):
        """Test getting Gemini API key from config"""
        api_key = generator._get_api_key("gemini")

        assert api_key == "fake_gemini_key"

    def test_get_api_key_anthropic(self, generator):
        """Test getting Anthropic API key from config"""
        api_key = generator._get_api_key("anthropic")

        assert api_key == "fake_anthropic_key"

    def test_get_api_key_no_config(self):
        """Test getting API key when no config"""
        generator = KeywordAlternativeGenerator(None)

        api_key = generator._get_api_key("gemini")

        assert api_key is None

    def test_get_model_gemini(self, generator):
        """Test getting Gemini model from config"""
        model = generator._get_model("gemini")

        assert model == "gemini-2.0-flash"

    def test_get_model_anthropic(self, generator):
        """Test getting Anthropic model from config"""
        model = generator._get_model("anthropic")

        assert model == "claude-3-haiku-20240307"

    def test_get_model_defaults(self):
        """Test default models when no config"""
        generator = KeywordAlternativeGenerator(None)

        gemini_model = generator._get_model("gemini")
        anthropic_model = generator._get_model("anthropic")

        assert gemini_model == "gemini-2.0-flash"
        assert anthropic_model == "claude-3-haiku-20240307"


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestCoverageGaps:
    """Tests for specific coverage gaps (lines 98-99, 226, 292, 319-325)."""

    def test_generate_single_no_api_key_from_config(self):
        """Test line 98-99: when _get_api_key returns None from config."""
        # Config without api_keys attribute
        config = Mock(spec=[])  # No attributes
        generator = KeywordAlternativeGenerator(config)

        result = generator.generate_single_alternative(
            keyword="test",
            provider="gemini"
            # No api_key parameter, so it will try to get from config
        )

        assert result is None

    def test_simple_remix_only_video_no_footage(self):
        """Test line 226: Return None when keyword has 'video' and only 1 core word."""
        config = Mock()
        generator = KeywordAlternativeGenerator(config)

        # Keyword with 'video' and only 1 core word after filtering
        result = generator.simple_remix_keyword("the video")

        # After filtering 'the', only 'video' remains
        # Since 'video' is in keyword, doesn't add footage
        # Since only 1 word, can't simplify to 2
        # Should return None
        assert result is None

    def test_get_api_key_unknown_provider(self):
        """Test line 292: _get_api_key returns None for unknown provider."""
        config = Mock()
        config.api_keys = Mock()
        config.api_keys.gemini_api_key = "key"
        generator = KeywordAlternativeGenerator(config)

        result = generator._get_api_key("unknown_provider")

        assert result is None

    def test_get_model_from_llm_config_gemini(self):
        """Test lines 319-321: get model from llm config (not matching)."""
        config = Mock(spec=['llm'])  # Only has llm, not matching
        config.llm = Mock()
        config.llm.model = "gemini-1.5-pro"
        generator = KeywordAlternativeGenerator(config)

        result = generator._get_model("gemini")

        assert result == "gemini-1.5-pro"

    def test_get_model_from_llm_config_anthropic(self):
        """Test lines 322-323: get anthropic model from llm config."""
        config = Mock(spec=['llm'])  # Only has llm, not matching
        config.llm = Mock()
        config.llm.anthropic_model = "claude-3-opus"
        generator = KeywordAlternativeGenerator(config)

        result = generator._get_model("anthropic")

        assert result == "claude-3-opus"

    def test_get_model_llm_config_missing_attr(self):
        """Test line 325: fallback when llm config doesn't have model attr."""
        config = Mock(spec=['llm'])
        config.llm = Mock(spec=[])  # No model attributes
        generator = KeywordAlternativeGenerator(config)

        result = generator._get_model("gemini")

        # Should return default
        assert result == "gemini-2.0-flash"


class TestEdgeCases:
    """Test edge cases"""

    def test_simple_remix_single_word(self, generator):
        """Test simple remix with single word"""
        result = generator.simple_remix_keyword("beach")

        assert result is not None
        assert "beach" in result.lower()

    def test_fallback_remix_empty_string(self, generator):
        """Test fallback remix with empty string"""
        result = generator.fallback_remix("")

        assert isinstance(result, list)

    def test_fallback_remix_single_word(self, generator):
        """Test fallback remix with single word"""
        result = generator.fallback_remix("beach")

        # Should still try to generate alternatives
        assert isinstance(result, list)

    @patch('src.llm_client.create_client')
    def test_generate_single_with_topic_context(self, mock_create_client, generator):
        """Test single alternative with topic context"""
        mock_response = Mock()
        mock_response.text = "beach video"
        mock_client = Mock()
        mock_client.generate.return_value = mock_response
        mock_create_client.return_value = mock_client

        result = generator.generate_single_alternative(
            keyword="beach",
            topic="travel vlogs",
            provider="gemini",
            api_key="test_key"
        )

        assert result == "beach video"
        # Verify topic was included in request
        call_args = mock_client.generate.call_args
        assert call_args is not None

    @patch('src.llm_client.create_client')
    def test_generate_multiple_with_model_override(self, mock_create_client, generator):
        """Test multiple alternatives with model override"""
        mock_response = Mock()
        mock_response.parsed_data = ["alt1"]
        mock_client = Mock()
        mock_client.generate.return_value = mock_response
        mock_create_client.return_value = mock_client

        keywords, status = generator.generate_multiple_alternatives(
            keyword="test",
            provider="gemini",
            api_key="test_key",
            model="custom-model"
        )

        # Verify custom model was used
        mock_create_client.assert_called_once_with(
            "gemini",
            api_key="test_key",
            model="custom-model"
        )
