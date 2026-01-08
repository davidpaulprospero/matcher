"""
Tests for keyword_extractor.models

Tests cover:
- KeywordResult dataclass creation and methods
- PrioritizedKeyword dataclass creation
- get_sorted_keywords() sorting logic
"""

import pytest
from src.keyword_extractor.models import KeywordResult, PrioritizedKeyword


class TestPrioritizedKeyword:
    """Test PrioritizedKeyword dataclass"""

    def test_create_prioritized_keyword(self):
        """Test creating PrioritizedKeyword with all fields"""
        pk = PrioritizedKeyword(
            keyword="mountain climbing",
            priority=0.9,
            source="entity",
            mention_count=5
        )
        assert pk.keyword == "mountain climbing"
        assert pk.priority == 0.9
        assert pk.source == "entity"
        assert pk.mention_count == 5

    def test_prioritized_keyword_defaults(self):
        """Test PrioritizedKeyword with default mention_count"""
        pk = PrioritizedKeyword(
            keyword="test",
            priority=0.5,
            source="general"
        )
        assert pk.mention_count == 1  # Default

    def test_prioritized_keyword_sources(self):
        """Test different source types"""
        sources = ["entity", "topic", "general", "list_item"]
        for source in sources:
            pk = PrioritizedKeyword(
                keyword="test",
                priority=0.5,
                source=source
            )
            assert pk.source == source


class TestKeywordResult:
    """Test KeywordResult dataclass"""

    def test_create_minimal_result(self):
        """Test creating KeywordResult with minimal fields"""
        result = KeywordResult(
            keywords=["keyword1", "keyword2"],
            segments_analyzed=5,
            extraction_method="llm"
        )
        assert result.keywords == ["keyword1", "keyword2"]
        assert result.segments_analyzed == 5
        assert result.extraction_method == "llm"
        assert result.entities == []  # Default
        assert result.topic == ""  # Default
        assert result.prioritized_keywords == []  # Default

    def test_create_full_result(self):
        """Test creating KeywordResult with all fields"""
        entities = [{"name": "John Doe", "type": "PERSON"}]
        prioritized = [
            PrioritizedKeyword("kw1", 0.9, "entity", 5),
            PrioritizedKeyword("kw2", 0.5, "general", 2)
        ]
        result = KeywordResult(
            keywords=["kw1", "kw2"],
            segments_analyzed=10,
            extraction_method="llm_entity_aware",
            entities=entities,
            topic="Documentary",
            prioritized_keywords=prioritized
        )
        assert result.entities == entities
        assert result.topic == "Documentary"
        assert len(result.prioritized_keywords) == 2

    def test_get_sorted_keywords_with_prioritized(self):
        """Test get_sorted_keywords() sorts by priority"""
        prioritized = [
            PrioritizedKeyword("low", 0.3, "general", 1),
            PrioritizedKeyword("high", 0.9, "entity", 5),
            PrioritizedKeyword("medium", 0.6, "topic", 3)
        ]
        result = KeywordResult(
            keywords=["low", "high", "medium"],
            segments_analyzed=5,
            extraction_method="llm",
            prioritized_keywords=prioritized
        )
        sorted_kw = result.get_sorted_keywords()
        assert sorted_kw == ["high", "medium", "low"]

    def test_get_sorted_keywords_without_prioritized(self):
        """Test get_sorted_keywords() returns original order without prioritized"""
        result = KeywordResult(
            keywords=["first", "second", "third"],
            segments_analyzed=5,
            extraction_method="tfidf"
        )
        sorted_kw = result.get_sorted_keywords()
        assert sorted_kw == ["first", "second", "third"]

    def test_get_sorted_keywords_empty(self):
        """Test get_sorted_keywords() with empty keywords"""
        result = KeywordResult(
            keywords=[],
            segments_analyzed=0,
            extraction_method="none"
        )
        sorted_kw = result.get_sorted_keywords()
        assert sorted_kw == []

    def test_extraction_methods(self):
        """Test different extraction method values"""
        methods = ["llm", "llm_entity_aware", "tfidf", "none"]
        for method in methods:
            result = KeywordResult(
                keywords=[],
                segments_analyzed=0,
                extraction_method=method
            )
            assert result.extraction_method == method
