"""
Tests for entity match scoring boost functionality (US-009).

Tests that matching named entities between voiceover and video segments
correctly boosts confidence with graduated values.
"""

import pytest
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import SRTSegment
from src.matching.scoring import (
    apply_entity_match_boost,
    _extract_entity_texts,
    _looks_like_entity
)

pytestmark = pytest.mark.unit


class TestApplyEntityMatchBoostFunction:
    """Tests for apply_entity_match_boost() function existence and signature."""

    @pytest.mark.fast
    def test_function_exists(self):
        """Function should exist in scoring module."""
        from src.matching import scoring
        assert hasattr(scoring, 'apply_entity_match_boost')

    @pytest.mark.fast
    def test_function_is_callable(self):
        """Function should be callable."""
        assert callable(apply_entity_match_boost)

    @pytest.mark.fast
    def test_function_accepts_required_args(self):
        """Function should accept confidence, vo_segment, video_segment."""
        vo = SRTSegment(index=0, start_time=0, end_time=5, text="test")
        video = SRTSegment(index=0, start_time=0, end_time=5, text="test")
        result = apply_entity_match_boost(0.7, vo, video)
        assert result is not None

    @pytest.mark.fast
    def test_function_returns_tuple(self):
        """Function should return a tuple."""
        vo = SRTSegment(index=0, start_time=0, end_time=5, text="test")
        video = SRTSegment(index=0, start_time=0, end_time=5, text="test")
        result = apply_entity_match_boost(0.7, vo, video)
        assert isinstance(result, tuple)
        assert len(result) == 3

    @pytest.mark.fast
    def test_return_types(self):
        """Function should return (float, str, list)."""
        vo = SRTSegment(
            index=0, start_time=0, end_time=5, text="test",
            entities=[{"text": "John Smith", "type": "PERSON"}]
        )
        video = SRTSegment(
            index=0, start_time=0, end_time=5, text="test",
            entities=[{"text": "John Smith", "type": "PERSON"}]
        )
        boosted, reason, entities = apply_entity_match_boost(0.7, vo, video)
        assert isinstance(boosted, float)
        assert isinstance(reason, str)
        assert isinstance(entities, list)


class TestGraduatedBoostValues:
    """Tests for graduated boost values based on entity match count."""

    @pytest.mark.fast
    def test_one_entity_match_boost_005(self):
        """One matching entity should give +0.05 boost."""
        vo = SRTSegment(
            index=0, start_time=0, end_time=5, text="test",
            entities=[{"text": "Barack Obama", "type": "PERSON"}]
        )
        video = SRTSegment(
            index=0, start_time=0, end_time=5, text="test",
            entities=[{"text": "Barack Obama", "type": "PERSON"}]
        )
        boosted, reason, entities = apply_entity_match_boost(0.70, vo, video)
        assert boosted == pytest.approx(0.75, abs=0.001)
        assert "+0.05" in reason

    @pytest.mark.fast
    def test_two_entity_match_boost_008(self):
        """Two matching entities should give +0.08 boost."""
        vo = SRTSegment(
            index=0, start_time=0, end_time=5, text="test",
            entities=[
                {"text": "Barack Obama", "type": "PERSON"},
                {"text": "White House", "type": "LOCATION"}
            ]
        )
        video = SRTSegment(
            index=0, start_time=0, end_time=5, text="test",
            entities=[
                {"text": "Barack Obama", "type": "PERSON"},
                {"text": "White House", "type": "LOCATION"}
            ]
        )
        boosted, reason, entities = apply_entity_match_boost(0.70, vo, video)
        assert boosted == pytest.approx(0.78, abs=0.001)
        assert "+0.08" in reason

    @pytest.mark.fast
    def test_three_plus_entity_match_boost_012(self):
        """Three or more matching entities should give +0.12 boost."""
        vo = SRTSegment(
            index=0, start_time=0, end_time=5, text="test",
            entities=[
                {"text": "Barack Obama", "type": "PERSON"},
                {"text": "White House", "type": "LOCATION"},
                {"text": "Democratic Party", "type": "ORG"}
            ]
        )
        video = SRTSegment(
            index=0, start_time=0, end_time=5, text="test",
            entities=[
                {"text": "Barack Obama", "type": "PERSON"},
                {"text": "White House", "type": "LOCATION"},
                {"text": "Democratic Party", "type": "ORG"}
            ]
        )
        boosted, reason, entities = apply_entity_match_boost(0.70, vo, video)
        assert boosted == pytest.approx(0.82, abs=0.001)
        assert "+0.12" in reason

    @pytest.mark.fast
    def test_four_entities_still_012_boost(self):
        """Four matching entities should still give +0.12 boost (max)."""
        vo = SRTSegment(
            index=0, start_time=0, end_time=5, text="test",
            entities=[
                {"text": "Barack Obama", "type": "PERSON"},
                {"text": "White House", "type": "LOCATION"},
                {"text": "Democratic Party", "type": "ORG"},
                {"text": "Washington DC", "type": "LOCATION"}
            ]
        )
        video = SRTSegment(
            index=0, start_time=0, end_time=5, text="test",
            entities=[
                {"text": "Barack Obama", "type": "PERSON"},
                {"text": "White House", "type": "LOCATION"},
                {"text": "Democratic Party", "type": "ORG"},
                {"text": "Washington DC", "type": "LOCATION"}
            ]
        )
        boosted, reason, entities = apply_entity_match_boost(0.70, vo, video)
        assert boosted == pytest.approx(0.82, abs=0.001)
        assert "+0.12" in reason


class TestNoBoostScenarios:
    """Tests for scenarios where no boost should be applied."""

    @pytest.mark.fast
    def test_no_entities_in_voiceover(self):
        """No boost when voiceover has no entities."""
        vo = SRTSegment(index=0, start_time=0, end_time=5, text="test", entities=[])
        video = SRTSegment(
            index=0, start_time=0, end_time=5, text="test",
            entities=[{"text": "Barack Obama", "type": "PERSON"}]
        )
        boosted, reason, entities = apply_entity_match_boost(0.70, vo, video)
        assert boosted == 0.70
        assert reason == ""
        assert entities == []

    @pytest.mark.fast
    def test_no_entities_in_video(self):
        """No boost when video has no entities."""
        vo = SRTSegment(
            index=0, start_time=0, end_time=5, text="test",
            entities=[{"text": "Barack Obama", "type": "PERSON"}]
        )
        video = SRTSegment(index=0, start_time=0, end_time=5, text="test", entities=[])
        boosted, reason, entities = apply_entity_match_boost(0.70, vo, video)
        assert boosted == 0.70
        assert reason == ""
        assert entities == []

    @pytest.mark.fast
    def test_no_matching_entities(self):
        """No boost when entities don't match."""
        vo = SRTSegment(
            index=0, start_time=0, end_time=5, text="test",
            entities=[{"text": "Barack Obama", "type": "PERSON"}]
        )
        video = SRTSegment(
            index=0, start_time=0, end_time=5, text="test",
            entities=[{"text": "Donald Trump", "type": "PERSON"}]
        )
        boosted, reason, entities = apply_entity_match_boost(0.70, vo, video)
        assert boosted == 0.70
        assert reason == ""
        assert entities == []

    @pytest.mark.fast
    def test_none_entities_field(self):
        """No boost when entities field is None."""
        vo = SRTSegment(index=0, start_time=0, end_time=5, text="test")
        vo.entities = None
        video = SRTSegment(index=0, start_time=0, end_time=5, text="test")
        video.entities = None
        boosted, reason, entities = apply_entity_match_boost(0.70, vo, video)
        assert boosted == 0.70


class TestCaseInsensitiveMatching:
    """Tests for case-insensitive entity matching."""

    @pytest.mark.fast
    def test_lowercase_match(self):
        """Matching should be case-insensitive (lowercase in video)."""
        vo = SRTSegment(
            index=0, start_time=0, end_time=5, text="test",
            entities=[{"text": "Barack Obama", "type": "PERSON"}]
        )
        video = SRTSegment(
            index=0, start_time=0, end_time=5, text="test",
            entities=[{"text": "barack obama", "type": "PERSON"}]
        )
        boosted, reason, entities = apply_entity_match_boost(0.70, vo, video)
        assert boosted > 0.70
        assert len(entities) == 1

    @pytest.mark.fast
    def test_uppercase_match(self):
        """Matching should be case-insensitive (uppercase in video)."""
        vo = SRTSegment(
            index=0, start_time=0, end_time=5, text="test",
            entities=[{"text": "barack obama", "type": "PERSON"}]
        )
        video = SRTSegment(
            index=0, start_time=0, end_time=5, text="test",
            entities=[{"text": "BARACK OBAMA", "type": "PERSON"}]
        )
        boosted, reason, entities = apply_entity_match_boost(0.70, vo, video)
        assert boosted > 0.70


class TestConfidenceCapping:
    """Tests for confidence capping at 1.0."""

    @pytest.mark.fast
    def test_boost_capped_at_one(self):
        """Confidence should not exceed 1.0 after boost."""
        vo = SRTSegment(
            index=0, start_time=0, end_time=5, text="test",
            entities=[
                {"text": "Barack Obama", "type": "PERSON"},
                {"text": "White House", "type": "LOCATION"},
                {"text": "Democratic Party", "type": "ORG"}
            ]
        )
        video = SRTSegment(
            index=0, start_time=0, end_time=5, text="test",
            entities=[
                {"text": "Barack Obama", "type": "PERSON"},
                {"text": "White House", "type": "LOCATION"},
                {"text": "Democratic Party", "type": "ORG"}
            ]
        )
        boosted, reason, entities = apply_entity_match_boost(0.95, vo, video)
        assert boosted == 1.0

    @pytest.mark.fast
    def test_already_at_one(self):
        """Starting at 1.0 should stay at 1.0."""
        vo = SRTSegment(
            index=0, start_time=0, end_time=5, text="test",
            entities=[{"text": "Barack Obama", "type": "PERSON"}]
        )
        video = SRTSegment(
            index=0, start_time=0, end_time=5, text="test",
            entities=[{"text": "Barack Obama", "type": "PERSON"}]
        )
        boosted, reason, entities = apply_entity_match_boost(1.0, vo, video)
        assert boosted == 1.0


class TestMatchedEntitiesReturn:
    """Tests for matched entities list in return value."""

    @pytest.mark.fast
    def test_returns_matched_entity_names(self):
        """Should return list of matched entity names."""
        vo = SRTSegment(
            index=0, start_time=0, end_time=5, text="test",
            entities=[{"text": "Barack Obama", "type": "PERSON"}]
        )
        video = SRTSegment(
            index=0, start_time=0, end_time=5, text="test",
            entities=[{"text": "Barack Obama", "type": "PERSON"}]
        )
        boosted, reason, entities = apply_entity_match_boost(0.70, vo, video)
        assert "Barack Obama" in entities

    @pytest.mark.fast
    def test_returns_multiple_matched_entities(self):
        """Should return all matched entity names."""
        vo = SRTSegment(
            index=0, start_time=0, end_time=5, text="test",
            entities=[
                {"text": "Barack Obama", "type": "PERSON"},
                {"text": "White House", "type": "LOCATION"}
            ]
        )
        video = SRTSegment(
            index=0, start_time=0, end_time=5, text="test",
            entities=[
                {"text": "Barack Obama", "type": "PERSON"},
                {"text": "White House", "type": "LOCATION"}
            ]
        )
        boosted, reason, entities = apply_entity_match_boost(0.70, vo, video)
        assert len(entities) == 2

    @pytest.mark.fast
    def test_reason_includes_entity_names(self):
        """Reason string should include matched entity names."""
        vo = SRTSegment(
            index=0, start_time=0, end_time=5, text="test",
            entities=[{"text": "Barack Obama", "type": "PERSON"}]
        )
        video = SRTSegment(
            index=0, start_time=0, end_time=5, text="test",
            entities=[{"text": "Barack Obama", "type": "PERSON"}]
        )
        boosted, reason, entities = apply_entity_match_boost(0.70, vo, video)
        assert "Barack Obama" in reason
        assert "entity match" in reason


class TestExtractEntityTextsHelper:
    """Tests for _extract_entity_texts() helper function."""

    @pytest.mark.fast
    def test_extracts_from_dict_entities(self):
        """Should extract text from entity dicts."""
        segment = SRTSegment(
            index=0, start_time=0, end_time=5, text="test",
            entities=[
                {"text": "John Smith", "type": "PERSON"},
                {"text": "New York", "type": "LOCATION"}
            ]
        )
        entities = _extract_entity_texts(segment)
        assert "John Smith" in entities
        assert "New York" in entities

    @pytest.mark.fast
    def test_extracts_from_string_entities(self):
        """Should extract from string entities (legacy format)."""
        segment = SRTSegment(
            index=0, start_time=0, end_time=5, text="test",
            entities=["John Smith", "New York"]
        )
        entities = _extract_entity_texts(segment)
        assert "John Smith" in entities
        assert "New York" in entities

    @pytest.mark.fast
    def test_skips_short_entities(self):
        """Should skip entities with < 2 characters."""
        segment = SRTSegment(
            index=0, start_time=0, end_time=5, text="test",
            entities=[{"text": "A", "type": "PERSON"}]
        )
        entities = _extract_entity_texts(segment)
        assert "A" not in entities

    @pytest.mark.fast
    def test_extracts_entity_like_keywords(self):
        """Should extract proper noun keywords as entities."""
        segment = SRTSegment(
            index=0, start_time=0, end_time=5, text="test",
            keywords=["John Smith", "climate change"],
            entities=[]
        )
        entities = _extract_entity_texts(segment)
        assert "John Smith" in entities
        # lowercase keyword should not be extracted as entity
        assert "climate change" not in entities

    @pytest.mark.fast
    def test_handles_empty_entities(self):
        """Should handle empty entities list."""
        segment = SRTSegment(
            index=0, start_time=0, end_time=5, text="test",
            entities=[]
        )
        entities = _extract_entity_texts(segment)
        assert entities == []

    @pytest.mark.fast
    def test_handles_none_entities(self):
        """Should handle None entities."""
        segment = SRTSegment(index=0, start_time=0, end_time=5, text="test")
        segment.entities = None
        entities = _extract_entity_texts(segment)
        assert entities == []


class TestLooksLikeEntityHelper:
    """Tests for _looks_like_entity() helper function."""

    @pytest.mark.fast
    def test_multi_word_capitalized_is_entity(self):
        """Multi-word capitalized phrases should be entities."""
        assert _looks_like_entity("John Smith") is True
        assert _looks_like_entity("New York City") is True
        assert _looks_like_entity("United States") is True

    @pytest.mark.fast
    def test_single_capitalized_word_is_entity(self):
        """Single capitalized word should be entity (if not common)."""
        assert _looks_like_entity("Obama") is True
        assert _looks_like_entity("Microsoft") is True

    @pytest.mark.fast
    def test_common_words_not_entities(self):
        """Common words should not be treated as entities."""
        assert _looks_like_entity("The") is False
        assert _looks_like_entity("Is") is False
        assert _looks_like_entity("They") is False

    @pytest.mark.fast
    def test_lowercase_not_entity(self):
        """Lowercase words should not be entities."""
        assert _looks_like_entity("climate change") is False
        assert _looks_like_entity("the president") is False

    @pytest.mark.fast
    def test_empty_string_not_entity(self):
        """Empty string should not be entity."""
        assert _looks_like_entity("") is False
        assert _looks_like_entity(None) is False

    @pytest.mark.fast
    def test_mixed_case_multi_word(self):
        """Mixed case multi-word should check majority capitalized."""
        # "United States of America" - 3 of 4 capitalized
        assert _looks_like_entity("United States of America") is True
        # "the United States" - starts with lowercase, so not detected as entity
        # (This is correct behavior - entities should start capitalized)
        assert _looks_like_entity("the United States") is False
        # "The United States" - starts with capital, detected as entity
        assert _looks_like_entity("The United States") is True


class TestPersonPlaceNameMatching:
    """Tests for matching person and place names (acceptance criteria)."""

    @pytest.mark.fast
    def test_person_name_match(self):
        """Matching person names should boost confidence."""
        vo = SRTSegment(
            index=0, start_time=0, end_time=5, text="Obama gave a speech",
            entities=[{"text": "Barack Obama", "type": "PERSON"}]
        )
        video = SRTSegment(
            index=0, start_time=0, end_time=5, text="video about Obama",
            entities=[{"text": "Barack Obama", "type": "PERSON"}]
        )
        boosted, reason, entities = apply_entity_match_boost(0.70, vo, video)
        assert boosted > 0.70
        assert "Barack Obama" in entities

    @pytest.mark.fast
    def test_place_name_match(self):
        """Matching place names should boost confidence."""
        vo = SRTSegment(
            index=0, start_time=0, end_time=5, text="in New York",
            entities=[{"text": "New York", "type": "LOCATION"}]
        )
        video = SRTSegment(
            index=0, start_time=0, end_time=5, text="New York skyline",
            entities=[{"text": "New York", "type": "LOCATION"}]
        )
        boosted, reason, entities = apply_entity_match_boost(0.70, vo, video)
        assert boosted > 0.70
        assert "New York" in entities

    @pytest.mark.fast
    def test_organization_name_match(self):
        """Matching organization names should boost confidence."""
        vo = SRTSegment(
            index=0, start_time=0, end_time=5, text="at Google HQ",
            entities=[{"text": "Google", "type": "ORG"}]
        )
        video = SRTSegment(
            index=0, start_time=0, end_time=5, text="Google campus tour",
            entities=[{"text": "Google", "type": "ORG"}]
        )
        boosted, reason, entities = apply_entity_match_boost(0.70, vo, video)
        assert boosted > 0.70


class TestLogging:
    """Tests for debug logging behavior."""

    @pytest.mark.fast
    def test_logs_boost_application(self, caplog):
        """Should log when boost is applied."""
        vo = SRTSegment(
            index=0, start_time=0, end_time=5, text="test",
            entities=[{"text": "Barack Obama", "type": "PERSON"}]
        )
        video = SRTSegment(
            index=0, start_time=0, end_time=5, text="test",
            entities=[{"text": "Barack Obama", "type": "PERSON"}]
        )
        with caplog.at_level(logging.DEBUG):
            apply_entity_match_boost(0.70, vo, video)
        assert any("Entity match boost applied" in r.message for r in caplog.records)

    @pytest.mark.fast
    def test_no_log_when_no_boost(self, caplog):
        """Should not log when no boost applied."""
        vo = SRTSegment(index=0, start_time=0, end_time=5, text="test", entities=[])
        video = SRTSegment(index=0, start_time=0, end_time=5, text="test", entities=[])
        with caplog.at_level(logging.DEBUG):
            apply_entity_match_boost(0.70, vo, video)
        assert not any("Entity match boost applied" in r.message for r in caplog.records)
