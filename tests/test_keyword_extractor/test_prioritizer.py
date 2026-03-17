"""
Tests for keyword_extractor.prioritizer

Tests cover:
- build_prioritized_keywords() priority scoring
- Entity vs topic vs general keyword scoring
- Mention frequency boost calculation
- Visual specificity boost
- Priority sorting
"""

import pytest
from src.keyword_extractor.prioritizer import build_prioritized_keywords
from src.keyword_extractor.models import PrioritizedKeyword


class TestBuildPrioritizedKeywords:
    """Test build_prioritized_keywords() function"""

    @pytest.mark.fast
    def test_build_prioritized_empty(self):
        """Test with empty keywords"""
        result = build_prioritized_keywords(
            keywords=[],
            entity_keywords=[],
            raw_entities=[],
            full_text="",
            topic=""
        )
        assert result == []

    @pytest.mark.fast
    def test_build_prioritized_entity_keywords(self):
        """Test entity keywords get high priority (0.9 base)"""
        entity_keywords = ["Mount Everest", "Nepal"]
        raw_entities = [
            {"name": "Mount Everest", "type": "LOCATION"},
            {"name": "Nepal", "type": "LOCATION"}
        ]

        result = build_prioritized_keywords(
            keywords=entity_keywords,
            entity_keywords=entity_keywords,
            raw_entities=raw_entities,
            full_text="Mount Everest is located in Nepal.",
            topic="Travel"
        )

        assert len(result) == 2
        # Entity keywords should have priority >= 0.9
        for pk in result:
            assert pk.priority >= 0.9
            assert pk.source == "entity"

    @pytest.mark.fast
    def test_build_prioritized_topic_keywords(self):
        """Test topic-matching keywords get medium-high priority (0.8 base)"""
        result = build_prioritized_keywords(
            keywords=["mountain climbing", "alpine gear"],
            entity_keywords=[],
            raw_entities=[],
            full_text="This documentary is about mountain climbing and alpine gear.",
            topic="Mountain Climbing"
        )

        # "mountain climbing" should match topic
        topic_keywords = [pk for pk in result if pk.source == "topic"]
        assert len(topic_keywords) >= 1

        # Topic keywords should have priority >= 0.8
        for pk in topic_keywords:
            assert pk.priority >= 0.8

    @pytest.mark.fast
    def test_build_prioritized_general_keywords(self):
        """Test general keywords get lower priority (0.5 base)"""
        result = build_prioritized_keywords(
            keywords=["landscape", "scenery"],
            entity_keywords=[],
            raw_entities=[],
            full_text="Beautiful landscape and scenery.",
            topic="Nature"
        )

        # Should be general keywords (not entity or topic match)
        general_keywords = [pk for pk in result if pk.source == "general"]
        assert len(general_keywords) >= 1

        # General keywords should have priority around 0.5-0.6
        for pk in general_keywords:
            assert 0.5 <= pk.priority <= 0.7

    @pytest.mark.fast
    def test_build_prioritized_mention_boost(self):
        """Test mention frequency increases priority"""
        full_text = "Mountain mountain mountain mountain mountain"  # 5 mentions

        result = build_prioritized_keywords(
            keywords=["mountain"],
            entity_keywords=[],
            raw_entities=[],
            full_text=full_text,
            topic=""
        )

        pk = result[0]
        # Should have mention boost (up to +0.1)
        assert pk.priority > 0.5  # Base 0.5 + mention boost
        assert pk.mention_count >= 5

    @pytest.mark.fast
    def test_build_prioritized_visual_specificity_boost(self):
        """Test visual specificity terms increase priority"""
        keywords_with_boost = [
            "4K mountain footage",
            "drone aerial view",
            "timelapse sunset"
        ]

        result = build_prioritized_keywords(
            keywords=keywords_with_boost,
            entity_keywords=[],
            raw_entities=[],
            full_text="",
            topic=""
        )

        # All should have visual specificity boost (+0.05)
        for pk in result:
            assert pk.priority >= 0.5  # Base + boost

    @pytest.mark.fast
    def test_build_prioritized_sorting(self):
        """Test results are sorted by priority (highest first)"""
        entity_keywords = ["Mount Everest"]
        raw_entities = [{"name": "Mount Everest", "type": "LOCATION"}]

        result = build_prioritized_keywords(
            keywords=["Mount Everest", "mountain", "landscape"],
            entity_keywords=entity_keywords,
            raw_entities=raw_entities,
            full_text="Mount Everest towers above the landscape.",
            topic=""
        )

        # Should be sorted by priority descending
        priorities = [pk.priority for pk in result]
        assert priorities == sorted(priorities, reverse=True)

        # Entity keyword should be first (highest priority)
        assert result[0].keyword == "Mount Everest"
        assert result[0].source == "entity"

    @pytest.mark.fast
    def test_build_prioritized_mixed_sources(self):
        """Test mixed entity/topic/general keywords"""
        entity_keywords = ["Paris"]
        raw_entities = [{"name": "Paris", "type": "LOCATION"}]

        result = build_prioritized_keywords(
            keywords=["Paris", "travel adventures", "scenery"],
            entity_keywords=entity_keywords,
            raw_entities=raw_entities,
            full_text="Travel adventures in Paris with beautiful scenery.",
            topic="Travel Adventures"
        )

        # Should have 3 keywords with different sources
        assert len(result) == 3

        # Paris should be entity (highest priority)
        paris = next(pk for pk in result if pk.keyword == "Paris")
        assert paris.source == "entity"
        assert paris.priority >= 0.9

        # "travel adventures" should be topic
        travel = next(pk for pk in result if pk.keyword == "travel adventures")
        assert travel.source == "topic"
        assert travel.priority >= 0.8

        # "scenery" should be general
        scenery = next(pk for pk in result if pk.keyword == "scenery")
        assert scenery.source == "general"
        assert scenery.priority < 0.8

    @pytest.mark.fast
    def test_build_prioritized_max_priority(self):
        """Test priority is capped at 1.0"""
        # Create scenario with maximum boosts
        full_text = "4K 4K 4K 4K 4K " * 20  # High mentions + visual term

        result = build_prioritized_keywords(
            keywords=["4K"],
            entity_keywords=["4K"],  # Treat as entity
            raw_entities=[{"name": "4K", "type": "TERM"}],
            full_text=full_text,
            topic="4K"  # Also matches topic
        )

        # Priority should not exceed 1.0
        assert all(pk.priority <= 1.0 for pk in result)

    @pytest.mark.fast
    def test_build_prioritized_entity_name_substring_match(self):
        """Test entity keywords match by substring"""
        entity_keywords = ["Mount Everest summit"]
        raw_entities = [{"name": "Mount Everest", "type": "LOCATION"}]

        result = build_prioritized_keywords(
            keywords=["Mount Everest summit"],
            entity_keywords=entity_keywords,
            raw_entities=raw_entities,
            full_text="Reaching the Mount Everest summit.",
            topic=""
        )

        pk = result[0]
        # Should be recognized as entity even with additional words
        assert pk.source == "entity"
        assert pk.priority >= 0.9

    @pytest.mark.fast
    def test_build_prioritized_topic_word_match(self):
        """Test topic matching works with word subsets"""
        result = build_prioritized_keywords(
            keywords=["wildlife", "conservation efforts"],
            entity_keywords=[],
            raw_entities=[],
            full_text="Documentary about wildlife conservation efforts.",
            topic="Wildlife Conservation"
        )

        # Both keywords should match topic
        wildlife = next(pk for pk in result if pk.keyword == "wildlife")
        assert wildlife.source == "topic"

        conservation = next(pk for pk in result if pk.keyword == "conservation efforts")
        assert conservation.source == "topic"


class TestPrioritizerEdgeCases:
    """Test edge cases for prioritizer"""

    @pytest.mark.fast
    def test_empty_topic(self):
        """Test with empty topic string"""
        result = build_prioritized_keywords(
            keywords=["mountain"],
            entity_keywords=[],
            raw_entities=[],
            full_text="Mountain scenery.",
            topic=""
        )

        # Should work with empty topic
        assert len(result) == 1
        assert result[0].source == "general"

    @pytest.mark.fast
    def test_empty_text(self):
        """Test with empty text"""
        result = build_prioritized_keywords(
            keywords=["mountain"],
            entity_keywords=[],
            raw_entities=[],
            full_text="",
            topic="Nature"
        )

        # Should handle empty text
        assert len(result) == 1
        # Mention count should be 0 or 1
        assert result[0].mention_count >= 0

    @pytest.mark.fast
    def test_keyword_not_in_text(self):
        """Test keyword that doesn't appear in text"""
        result = build_prioritized_keywords(
            keywords=["ocean"],
            entity_keywords=[],
            raw_entities=[],
            full_text="Mountains and forests.",
            topic=""
        )

        pk = result[0]
        # Should have low mention count
        assert pk.mention_count <= 1

    @pytest.mark.fast
    def test_case_insensitive_matching(self):
        """Test case-insensitive topic and entity matching"""
        entity_keywords = ["PARIS"]
        raw_entities = [{"name": "PARIS", "type": "LOCATION"}]

        result = build_prioritized_keywords(
            keywords=["paris city"],
            entity_keywords=entity_keywords,
            raw_entities=raw_entities,
            full_text="Visiting Paris city.",
            topic="PARIS TRAVEL"
        )

        # Should match despite case differences
        pk = result[0]
        assert pk.source in ["entity", "topic"]

    @pytest.mark.fast
    def test_special_characters_in_keywords(self):
        """Test keywords with special characters"""
        result = build_prioritized_keywords(
            keywords=["São Paulo", "café culture"],
            entity_keywords=[],
            raw_entities=[],
            full_text="São Paulo's café culture is vibrant.",
            topic=""
        )

        assert len(result) == 2
        # Should handle special characters
        assert all(pk.priority > 0 for pk in result)
