"""
Tests for keyword_extractor.core

Integration tests covering:
- LLMKeywordExtractor initialization
- extract_keywords() with mocked LLM
- extract_keyword_per_segment() integration
- TF-IDF fallback when no LLM available
- Footage suffix addition
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.keyword_extractor.core import LLMKeywordExtractor
from src.keyword_extractor.models import KeywordResult


@pytest.fixture
def mock_config():
    """Create mock configuration"""
    config = Mock()

    # LLM config
    config.llm = Mock()
    config.llm.provider = 'anthropic'
    config.llm.api_key = 'test-key'
    config.llm.model = 'claude-3-haiku-20240307'
    config.llm.max_tokens = 2000

    # Keyword config
    config.keyword = Mock()
    config.keyword.max_keywords = 30
    config.keyword.max_keyword_words = 8  # Default value for visual keyword validation

    # Keyword weights config (for TF-IDF fallback)
    config.keyword_weights = Mock()
    config.keyword_weights.auto_detect_count = 30

    return config


class TestLLMKeywordExtractorInit:
    """Test LLMKeywordExtractor initialization"""

    def test_init_with_anthropic(self, mock_config):
        """Test initialization with Anthropic provider"""
        with patch('src.llm_client.create_client') as mock_create:
            mock_client = Mock()
            mock_create.return_value = mock_client

            extractor = LLMKeywordExtractor(mock_config)

            assert extractor.config == mock_config
            assert extractor.llm_client == mock_client
            assert extractor.llm_provider == 'anthropic'

    def test_init_with_gemini(self, mock_config):
        """Test initialization with Gemini provider"""
        mock_config.llm.provider = 'google'

        with patch('src.llm_client.create_client') as mock_create:
            mock_client = Mock()
            mock_create.return_value = mock_client

            extractor = LLMKeywordExtractor(mock_config)

            assert extractor.llm_provider == 'google'

    def test_init_without_api_key(self, mock_config):
        """Test initialization without API key"""
        mock_config.llm.api_key = None

        with patch.dict('os.environ', {}, clear=True):
            with patch('src.llm_client.create_client') as mock_create:
                extractor = LLMKeywordExtractor(mock_config)

                # Should warn but not crash
                assert extractor.llm_client is None or mock_create.call_count == 0

    def test_init_handles_import_error(self, mock_config):
        """Test initialization handles import errors gracefully"""
        with patch('src.llm_client.create_client', side_effect=ImportError("No module")):
            extractor = LLMKeywordExtractor(mock_config)

            # Should handle gracefully
            assert extractor.config == mock_config


class TestExtractKeywordsWithLLM:
    """Test extract_keywords() with LLM"""

    def test_extract_keywords_basic(self, mock_config):
        """Test basic keyword extraction"""
        with patch('src.llm_client.create_client'):
            extractor = LLMKeywordExtractor(mock_config)

            # Mock LLM call to return keywords
            extractor._call_llm = Mock(side_effect=[
                '[]',  # entities
                '["mountain climbing", "alpine peaks", "snow"]',  # keywords
                '["mountain climbing", "alpine peaks", "snow", "expedition"]'  # expansion
            ])

            segments = [
                {'text': 'This documentary is about mountain climbing.'},
                {'text': 'We scaled alpine peaks covered in snow.'}
            ]

            result = extractor.extract_keywords(segments, max_keywords=30)

            assert isinstance(result, KeywordResult)
            assert len(result.keywords) > 0
            assert result.segments_analyzed == 2
            assert result.extraction_method == "llm_entity_aware"

    def test_extract_keywords_with_entities(self, mock_config):
        """Test keyword extraction with entity extraction"""
        with patch('src.llm_client.create_client'):
            extractor = LLMKeywordExtractor(mock_config)

            # Mock LLM client for topic detection
            mock_topic_response = Mock()
            mock_topic_response.text = 'Mount Everest climbing'
            extractor.llm_client.generate = Mock(return_value=mock_topic_response)

            # Mock _call_llm for entity extraction and keyword extraction
            extractor._call_llm = Mock(side_effect=[
                '{"places": [{"name": "Mount Everest", "search_keyword": "Mount Everest climbing", "context": "climbing"}]}',  # entities
                '["climbing", "expedition"]',  # keywords
                '["Mount Everest climbing", "climbing", "expedition"]'  # expansion
            ])

            segments = [{'text': 'We climbed Mount Everest.'}]

            result = extractor.extract_keywords(segments)

            # Should have entities
            assert len(result.entities) > 0
            assert result.entities[0]['text'] == 'Mount Everest'

            # Should have entity-based keywords
            assert any('Everest' in kw for kw in result.keywords)

    def test_extract_keywords_no_expand(self, mock_config):
        """Test keyword extraction without expansion"""
        with patch('src.llm_client.create_client'):
            extractor = LLMKeywordExtractor(mock_config)

            extractor._call_llm = Mock(side_effect=[
                '[]',  # entities
                '["keyword1", "keyword2"]'  # keywords
            ])

            segments = [{'text': 'Some text.'}]

            result = extractor.extract_keywords(segments, expand=False)

            # Should only call LLM twice (entities + keywords, no expansion)
            assert extractor._call_llm.call_count == 2

    def test_extract_keywords_empty_segments(self, mock_config):
        """Test keyword extraction with empty segments"""
        with patch('src.llm_client.create_client'):
            extractor = LLMKeywordExtractor(mock_config)

            result = extractor.extract_keywords([])

            assert result.keywords == []
            assert result.segments_analyzed == 0
            assert result.extraction_method == "none"

    def test_extract_keywords_max_keywords_limit(self, mock_config):
        """Test keyword extraction respects max_keywords"""
        with patch('src.llm_client.create_client'):
            extractor = LLMKeywordExtractor(mock_config)

            # Return many keywords
            many_keywords = [f"keyword{i}" for i in range(50)]
            extractor._call_llm = Mock(side_effect=[
                '[]',  # entities
                f'{many_keywords}',  # keywords
            ])

            segments = [{'text': 'Text with many concepts.'}]

            result = extractor.extract_keywords(segments, max_keywords=10, expand=False)

            # Should limit to max_keywords
            assert len(result.keywords) <= 10


class TestExtractKeywordsWithTFIDF:
    """Test extract_keywords() fallback to TF-IDF"""

    def test_extract_keywords_tfidf_fallback(self, mock_config):
        """Test TF-IDF fallback when no LLM client"""
        extractor = LLMKeywordExtractor(mock_config)
        extractor.llm_client = None  # Force TF-IDF fallback

        # Mock the KeywordWeightExtractor class that gets imported inside _extract_with_tfidf
        mock_extractor_instance = Mock()
        mock_extractor_instance.extract_weighted_terms = Mock(
            return_value=(['mountain', 'climbing'], [], {})
        )
        MockExtractor = Mock(return_value=mock_extractor_instance)

        # Mock the import statement
        import sys
        mock_module = Mock()
        mock_module.KeywordWeightExtractor = MockExtractor
        sys.modules['keyword_extractor'] = mock_module

        try:
            segments = [
                {'text': 'Mountain climbing is challenging.'},
                {'text': 'Climbing mountains requires skill.'}
            ]

            result = extractor.extract_keywords(segments, max_keywords=30)

            assert result.extraction_method == "tfidf"
            assert len(result.keywords) > 0
            assert 'mountain' in result.keywords or 'climbing' in result.keywords
        finally:
            # Clean up
            if 'keyword_extractor' in sys.modules:
                del sys.modules['keyword_extractor']

    def test_extract_keywords_tfidf_empty_segments(self, mock_config):
        """Test TF-IDF fallback with empty segments"""
        extractor = LLMKeywordExtractor(mock_config)
        extractor.llm_client = None

        result = extractor.extract_keywords([])

        assert result.keywords == []
        assert result.extraction_method == "none"


class TestExtractKeywordPerSegment:
    """Test extract_keyword_per_segment() wrapper"""

    def test_extract_keyword_per_segment_basic(self, mock_config):
        """Test per-segment keyword extraction"""
        with patch('src.llm_client.create_client'):
            extractor = LLMKeywordExtractor(mock_config)

            # Mock segment processor function
            with patch('src.keyword_extractor.core.extract_keyword_per_segment') as mock_extract:
                mock_extract.return_value = ["mountain", "ocean", "forest"]

                segments = [
                    {'text': 'Mountains are tall.'},
                    {'text': 'Oceans are deep.'},
                    {'text': 'Forests are green.'}
                ]

                result = extractor.extract_keyword_per_segment(segments, topic="Nature")

                assert result == ["mountain", "ocean", "forest"]

                # Verify correct parameters passed
                mock_extract.assert_called_once()
                call_kwargs = mock_extract.call_args[1]
                assert call_kwargs['topic'] == "Nature"


class TestAddFootageSuffixes:
    """Test add_footage_suffixes() method"""

    def test_add_footage_suffixes_default(self, mock_config):
        """Test adding default footage suffixes"""
        with patch('src.llm_client.create_client'):
            extractor = LLMKeywordExtractor(mock_config)

            keywords = ["mountain", "ocean"]

            result = extractor.add_footage_suffixes(keywords)

            # Should expand each keyword with suffixes
            assert len(result) > len(keywords)
            assert "mountain" in result
            assert "mountain 4K footage" in result or "mountain news footage" in result

    def test_add_footage_suffixes_custom(self, mock_config):
        """Test adding custom footage suffixes"""
        with patch('src.llm_client.create_client'):
            extractor = LLMKeywordExtractor(mock_config)

            keywords = ["mountain"]
            suffixes = ["drone footage", "aerial view"]

            result = extractor.add_footage_suffixes(keywords, suffixes=suffixes)

            assert "mountain" in result
            assert "mountain drone footage" in result
            assert "mountain aerial view" in result

    def test_add_footage_suffixes_empty(self, mock_config):
        """Test adding suffixes to empty list"""
        with patch('src.llm_client.create_client'):
            extractor = LLMKeywordExtractor(mock_config)

            result = extractor.add_footage_suffixes([])

            assert result == []


class TestLLMCallHelpers:
    """Test LLM call helper methods"""

    def test_call_llm_success(self, mock_config):
        """Test _call_llm() successful call"""
        with patch('src.llm_client.create_client'):
            extractor = LLMKeywordExtractor(mock_config)

            # Mock LLM client
            mock_client = Mock()
            mock_response = Mock()
            mock_response.text = '["keyword1", "keyword2"]'
            mock_client.generate = Mock(return_value=mock_response)
            extractor.llm_client = mock_client

            result = extractor._call_llm("Test prompt")

            assert result == '["keyword1", "keyword2"]'

    def test_call_llm_error(self, mock_config):
        """Test _call_llm() handles errors"""
        with patch('src.llm_client.create_client'):
            extractor = LLMKeywordExtractor(mock_config)

            mock_client = Mock()
            mock_client.generate = Mock(side_effect=Exception("API Error"))
            extractor.llm_client = mock_client

            result = extractor._call_llm("Test prompt")

            # Should return empty array on error
            assert result == "[]"

    def test_parse_keywords_json_valid(self, mock_config):
        """Test _parse_keywords_json() with valid JSON"""
        with patch('src.llm_client.create_client'):
            extractor = LLMKeywordExtractor(mock_config)

            response = '["keyword1", "keyword2", "keyword3"]'

            result = extractor._parse_keywords_json(response)

            assert result == ["keyword1", "keyword2", "keyword3"]

    def test_parse_keywords_json_with_extra_text(self, mock_config):
        """Test _parse_keywords_json() with extra text around JSON"""
        with patch('src.llm_client.create_client'):
            extractor = LLMKeywordExtractor(mock_config)

            response = 'Here are the keywords:\n["kw1", "kw2"]\nDone.'

            result = extractor._parse_keywords_json(response)

            assert result == ["kw1", "kw2"]

    def test_parse_keywords_json_fallback(self, mock_config):
        """Test _parse_keywords_json() fallback to line-by-line"""
        with patch('src.llm_client.create_client'):
            extractor = LLMKeywordExtractor(mock_config)

            response = """
            - keyword1
            - keyword2
            - keyword3
            """

            result = extractor._parse_keywords_json(response)

            # Should parse line by line
            assert len(result) > 0

    def test_parse_keywords_json_safety_limit(self, mock_config):
        """Test _parse_keywords_json() safety limit"""
        with patch('src.llm_client.create_client'):
            extractor = LLMKeywordExtractor(mock_config)

            # Create response with too many keywords
            many_keywords = ['\"kw' + str(i) + '\"' for i in range(100)]
            response = '[' + ', '.join(many_keywords) + ']'

            result = extractor._parse_keywords_json(response)

            # Should limit to 50
            assert len(result) <= 50


class TestCombineVoiceoverText:
    """Test _combine_voiceover_text() helper"""

    def test_combine_voiceover_text_basic(self, mock_config):
        """Test combining voiceover segments"""
        with patch('src.llm_client.create_client'):
            extractor = LLMKeywordExtractor(mock_config)

            segments = [
                {'text': 'First segment.'},
                {'text': 'Second segment.'},
                {'text': 'Third segment.'}
            ]

            result = extractor._combine_voiceover_text(segments)

            assert result == "First segment. Second segment. Third segment."

    def test_combine_voiceover_text_empty(self, mock_config):
        """Test combining empty segments"""
        with patch('src.llm_client.create_client'):
            extractor = LLMKeywordExtractor(mock_config)

            result = extractor._combine_voiceover_text([])

            assert result == ""

    def test_combine_voiceover_text_with_empty_text(self, mock_config):
        """Test combining segments with some empty text"""
        with patch('src.llm_client.create_client'):
            extractor = LLMKeywordExtractor(mock_config)

            segments = [
                {'text': 'First.'},
                {'text': ''},
                {'text': 'Third.'}
            ]

            result = extractor._combine_voiceover_text(segments)

            assert result == "First. Third."
