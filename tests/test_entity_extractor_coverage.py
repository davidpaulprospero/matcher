"""
Tests for src/keyword_extractor/entity_extractor.py coverage gaps.

Targets:
- Lines 85-90: Organization entity processing
- Lines 136-137: Exception handling
"""

import pytest
import json
from unittest.mock import MagicMock, patch

from src.keyword_extractor.entity_extractor import extract_entities, parse_entity_json


class TestExtractEntitiesOrganizations:
    """Test organization entity extraction (lines 85-90)."""

    @pytest.mark.fast
    def test_organizations_with_valid_keywords(self):
        """Test extracting organizations with valid search keywords."""
        response = json.dumps({
            "people": [],
            "places": [],
            "organizations": [
                {
                    "name": "United Nations",
                    "search_keyword": "UN headquarters building",
                    "context": "international organization"
                },
                {
                    "name": "NASA",
                    "search_keyword": "NASA space center",
                    "context": "space agency"
                }
            ],
            "dates": [],
            "events": []
        })

        mock_llm = MagicMock(return_value=response)
        keywords, entities = extract_entities("test text", "topic", mock_llm)

        assert "UN headquarters building" in keywords
        assert "NASA space center" in keywords
        assert len(entities) == 2
        assert entities[0]['type'] == 'ORG'
        assert entities[0]['text'] == 'United Nations'

    @pytest.mark.fast
    def test_organizations_with_short_keywords(self):
        """Test that short keywords (<=3 chars) are excluded."""
        response = json.dumps({
            "people": [],
            "places": [],
            "organizations": [
                {
                    "name": "FBI",
                    "search_keyword": "FBI",  # Too short
                    "context": "law enforcement"
                }
            ],
            "dates": [],
            "events": []
        })

        mock_llm = MagicMock(return_value=response)
        keywords, entities = extract_entities("test text", "topic", mock_llm)

        # FBI keyword too short, should not be in keywords
        assert "FBI" not in keywords
        # But entity should still be recorded
        assert len(entities) == 1
        assert entities[0]['text'] == 'FBI'

    @pytest.mark.fast
    def test_organizations_without_keywords(self):
        """Test organizations with names but empty keywords (line 89-90)."""
        response = json.dumps({
            "people": [],
            "places": [],
            "organizations": [
                {
                    "name": "World Health Organization",
                    "search_keyword": "",  # Empty keyword
                    "context": "health agency"
                }
            ],
            "dates": [],
            "events": []
        })

        mock_llm = MagicMock(return_value=response)
        keywords, entities = extract_entities("test text", "topic", mock_llm)

        # No keyword added but entity recorded
        assert len(keywords) == 0
        assert len(entities) == 1
        assert entities[0]['text'] == 'World Health Organization'
        assert entities[0]['search_keyword'] == ''


class TestExtractEntitiesExceptionHandling:
    """Test exception handling (lines 136-137)."""

    @pytest.mark.fast
    def test_llm_call_raises_exception(self):
        """Test handling when LLM call raises exception."""
        mock_llm = MagicMock(side_effect=Exception("API error"))

        with patch('src.keyword_extractor.entity_extractor.logger') as mock_logger:
            keywords, entities = extract_entities("test text", "topic", mock_llm)

            assert keywords == []
            assert entities == []
            mock_logger.warning.assert_called_once()
            assert "Entity extraction failed" in str(mock_logger.warning.call_args)

    @pytest.mark.fast
    def test_invalid_json_response(self):
        """Test handling when LLM returns invalid JSON."""
        mock_llm = MagicMock(return_value="not valid json at all")

        with patch('src.keyword_extractor.entity_extractor.logger') as mock_logger:
            keywords, entities = extract_entities("test text", "topic", mock_llm)

            assert keywords == []
            assert entities == []

    @pytest.mark.fast
    def test_malformed_json_structure(self):
        """Test handling when JSON has unexpected structure."""
        mock_llm = MagicMock(return_value='{"unexpected": "structure"}')

        keywords, entities = extract_entities("test text", "topic", mock_llm)

        # Should return empty but not crash
        assert keywords == []
        assert entities == []


class TestExtractEntitiesAllTypes:
    """Test all entity types for completeness."""

    @pytest.mark.fast
    def test_people_entities(self):
        """Test extracting people entities."""
        response = json.dumps({
            "people": [
                {
                    "name": "Albert Einstein",
                    "search_keyword": "Einstein portrait scientist",
                    "context": "physicist"
                }
            ],
            "places": [],
            "organizations": [],
            "dates": [],
            "events": []
        })

        mock_llm = MagicMock(return_value=response)
        keywords, entities = extract_entities("test text", "topic", mock_llm)

        assert "Einstein portrait scientist" in keywords
        assert entities[0]['type'] == 'PERSON'

    @pytest.mark.fast
    def test_places_entities(self):
        """Test extracting place entities."""
        response = json.dumps({
            "people": [],
            "places": [
                {
                    "name": "Paris",
                    "search_keyword": "Paris Eiffel Tower",
                    "context": "capital city"
                }
            ],
            "organizations": [],
            "dates": [],
            "events": []
        })

        mock_llm = MagicMock(return_value=response)
        keywords, entities = extract_entities("test text", "topic", mock_llm)

        assert "Paris Eiffel Tower" in keywords
        assert entities[0]['type'] == 'GPE'

    @pytest.mark.fast
    def test_dates_entities(self):
        """Test extracting date entities."""
        response = json.dumps({
            "people": [],
            "places": [],
            "organizations": [],
            "dates": [
                {
                    "date": "1969",
                    "search_keyword": "1969 moon landing",
                    "context": "historic year"
                }
            ],
            "events": []
        })

        mock_llm = MagicMock(return_value=response)
        keywords, entities = extract_entities("test text", "topic", mock_llm)

        assert "1969 moon landing" in keywords
        assert entities[0]['type'] == 'DATE'

    @pytest.mark.fast
    def test_events_entities(self):
        """Test extracting event entities."""
        response = json.dumps({
            "people": [],
            "places": [],
            "organizations": [],
            "dates": [],
            "events": [
                {
                    "name": "World War II",
                    "search_keyword": "WWII battlefield footage",
                    "context": "historic conflict"
                }
            ]
        })

        mock_llm = MagicMock(return_value=response)
        keywords, entities = extract_entities("test text", "topic", mock_llm)

        assert "WWII battlefield footage" in keywords
        assert entities[0]['type'] == 'EVENT'

    @pytest.mark.fast
    def test_mixed_entities(self):
        """Test extracting multiple entity types."""
        response = json.dumps({
            "people": [{"name": "JFK", "search_keyword": "JFK president speech", "context": ""}],
            "places": [{"name": "Dallas", "search_keyword": "Dallas Texas skyline", "context": ""}],
            "organizations": [{"name": "CIA", "search_keyword": "CIA headquarters Langley", "context": ""}],
            "dates": [{"date": "1963", "search_keyword": "1963 historic events", "context": ""}],
            "events": [{"name": "Assassination", "search_keyword": "historic assassination", "context": ""}]
        })

        mock_llm = MagicMock(return_value=response)
        keywords, entities = extract_entities("test text", "topic", mock_llm)

        assert len(keywords) == 5
        assert len(entities) == 5


class TestExtractEntitiesDeduplication:
    """Test keyword deduplication."""

    @pytest.mark.fast
    def test_duplicate_keywords_removed(self):
        """Test that duplicate keywords are deduplicated."""
        response = json.dumps({
            "people": [
                {"name": "Person1", "search_keyword": "same keyword here", "context": ""},
                {"name": "Person2", "search_keyword": "Same Keyword Here", "context": ""}  # Case insensitive dup
            ],
            "places": [],
            "organizations": [],
            "dates": [],
            "events": []
        })

        mock_llm = MagicMock(return_value=response)
        keywords, entities = extract_entities("test text", "topic", mock_llm)

        # Should only have one keyword (case-insensitive dedup)
        assert len(keywords) == 1
        # But both entities should be recorded
        assert len(entities) == 2


class TestParseEntityJson:
    """Test parse_entity_json function."""

    @pytest.mark.fast
    def test_parse_valid_json(self):
        """Test parsing valid JSON response."""
        response = '{"people": [], "places": []}'
        result = parse_entity_json(response)

        assert result == {"people": [], "places": []}

    @pytest.mark.fast
    def test_parse_json_with_surrounding_text(self):
        """Test parsing JSON with surrounding text."""
        response = 'Here is the response: {"data": "value"} end'
        result = parse_entity_json(response)

        assert result == {"data": "value"}

    @pytest.mark.fast
    def test_parse_empty_response(self):
        """Test parsing empty response."""
        result = parse_entity_json("")
        assert result == {}

    @pytest.mark.fast
    def test_parse_no_json(self):
        """Test parsing response with no JSON."""
        result = parse_entity_json("no json here")
        assert result == {}

    @pytest.mark.fast
    def test_parse_whitespace_response(self):
        """Test parsing whitespace-only response."""
        result = parse_entity_json("   \n\t  ")
        assert result == {}
