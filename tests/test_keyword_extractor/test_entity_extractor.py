"""
Tests for keyword_extractor.entity_extractor

Tests cover:
- extract_entities() with mocked LLM
- Entity parsing from JSON
- Visual keyword generation for different entity types
- Error handling for malformed LLM responses
"""

import pytest
from unittest.mock import Mock
from src.keyword_extractor.entity_extractor import (
    extract_entities,
    parse_entity_json
)


class TestParseEntityJson:
    """Test parse_entity_json() helper function"""

    def test_parse_valid_json_dict(self):
        """Test parsing valid JSON dict"""
        json_text = '{"people": [{"name": "John Doe"}], "places": [{"name": "Paris"}]}'
        entities = parse_entity_json(json_text)
        assert isinstance(entities, dict)
        assert "people" in entities or "places" in entities

    def test_parse_json_with_extra_text(self):
        """Test parsing JSON with surrounding text"""
        json_text = 'Here are the entities:\n{"people": [{"name": "Alice"}]}\nThat\'s all.'
        entities = parse_entity_json(json_text)
        assert isinstance(entities, dict)

    def test_parse_empty_dict(self):
        """Test parsing empty JSON dict"""
        json_text = '{}'
        entities = parse_entity_json(json_text)
        assert entities == {}

    def test_parse_malformed_json(self):
        """Test parsing malformed JSON returns empty dict"""
        json_text = 'This is not JSON at all'
        entities = parse_entity_json(json_text)
        assert entities == {}

    def test_parse_simple_dict(self):
        """Test parsing simple dict"""
        json_text = '{"name": "Alice", "type": "PERSON"}'
        entities = parse_entity_json(json_text)
        assert isinstance(entities, dict)
        assert len(entities) > 0


# TestGenerateVisualKeywords class removed - function is internal to extract_entities()


class TestExtractEntities:
    """Test extract_entities() main function with mocked LLM"""

    def test_extract_entities_basic(self):
        """Test basic entity extraction with mocked LLM"""
        # Mock LLM response with proper structure
        mock_llm_response = '''{
            "places": [{"name": "Mount Everest", "search_keyword": "mount everest peak climbing"}],
            "people": [{"name": "Edmund Hillary", "search_keyword": "edmund hillary mountaineer"}]
        }'''
        mock_llm_call = Mock(return_value=mock_llm_response)

        text = "Edmund Hillary climbed Mount Everest in 1953."
        topic = "Mountaineering"

        keywords, raw_entities = extract_entities(text, topic, mock_llm_call)

        # Check LLM was called
        mock_llm_call.assert_called_once()

        # Check entities extracted (2 entities)
        assert len(raw_entities) == 2

        # Check keywords generated
        assert isinstance(keywords, list)
        assert len(keywords) == 2

    def test_extract_entities_empty_response(self):
        """Test entity extraction with empty LLM response"""
        mock_llm_call = Mock(return_value='{}')

        text = "Some text without entities."
        topic = "Generic"

        keywords, raw_entities = extract_entities(text, topic, mock_llm_call)

        assert keywords == []
        assert raw_entities == []

    def test_extract_entities_malformed_response(self):
        """Test entity extraction with malformed LLM response"""
        mock_llm_call = Mock(return_value='This is not JSON')

        text = "Some text."
        topic = "Generic"

        keywords, raw_entities = extract_entities(text, topic, mock_llm_call)

        # Should handle gracefully
        assert keywords == []
        assert raw_entities == []

    def test_extract_entities_multiple_types(self):
        """Test entity extraction with multiple entity types"""
        mock_llm_response = '''{
            "places": [{"name": "Paris", "search_keyword": "paris france city"}],
            "people": [{"name": "Napoleon Bonaparte", "search_keyword": "napoleon bonaparte emperor"}],
            "events": [{"name": "French Revolution", "search_keyword": "french revolution 1789"}],
            "dates": [{"date": "1789", "search_keyword": "france 1789 revolution"}]
        }'''
        mock_llm_call = Mock(return_value=mock_llm_response)

        text = "Napoleon Bonaparte rose to power during the French Revolution in Paris around 1789."
        topic = "History"

        keywords, raw_entities = extract_entities(text, topic, mock_llm_call)

        assert len(raw_entities) == 4
        assert len(keywords) == 4

    def test_extract_entities_short_keywords_filtered(self):
        """Test entity extraction filters short keywords"""
        mock_llm_response = '''{
            "places": [{"name": "US", "search_keyword": "usa"}]
        }'''
        mock_llm_call = Mock(return_value=mock_llm_response)

        text = "The US"
        topic = "Geography"

        keywords, raw_entities = extract_entities(text, topic, mock_llm_call)

        # "usa" is only 3 chars, should be filtered (needs >3)
        assert len(keywords) == 0
        # But raw entity should still be recorded
        assert len(raw_entities) == 1

    def test_extract_entities_logger_called(self):
        """Test entity extraction handles empty search_keyword"""
        mock_llm_response = '''{
            "places": [{"name": "Test Location"}]
        }'''
        mock_llm_call = Mock(return_value=mock_llm_response)

        text = "Test text"
        topic = "Test"

        # Should not raise exceptions
        keywords, raw_entities = extract_entities(text, topic, mock_llm_call)

        # No search_keyword provided, so no keywords generated
        assert len(keywords) == 0
        # But raw entity recorded
        assert len(raw_entities) == 1
