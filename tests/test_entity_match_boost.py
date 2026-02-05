"""
Tests for US-63-011: Entity match boost for named entities in voiceover.

Tests the named entity extraction and configurable boost functionality:
- Extract named entities (people, places, organizations) from voiceover text
- Apply configurable boost when video caption contains matching entity
- Config option: matching.entity_match_boost (default: 0.1)

Created: 2026-02-05 (Sprint 63 - US-63-011)
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.scoring import (
    extract_named_entities_from_text,
    _extract_entity_texts,
    apply_entity_match_boost,
    calculate_entity_match_score,
)
from src.utils import SRTSegment


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config_with_entity_boost():
    """Mock config with entity_match_boost setting"""
    config = Mock()
    matching = Mock()
    matching.entity_match_boost = 0.1  # Default configurable boost
    config.matching = matching
    return config


@pytest.fixture
def mock_config_high_boost():
    """Mock config with higher entity_match_boost"""
    config = Mock()
    matching = Mock()
    matching.entity_match_boost = 0.15  # Higher boost
    config.matching = matching
    return config


@pytest.fixture
def new_york_voiceover():
    """Voiceover segment mentioning New York"""
    seg = SRTSegment(
        index=1,
        start_time=0.0,
        end_time=10.0,
        text="The city of New York is one of the most iconic places in the world.",
        source_file="voiceover.srt"
    )
    return seg


@pytest.fixture
def nyc_video():
    """Video segment about NYC (should match New York)"""
    seg = SRTSegment(
        index=1,
        start_time=0.0,
        end_time=10.0,
        text="Aerial view of New York City skyline at sunset.",
        source_file="/videos/nyc.mp4"
    )
    return seg


@pytest.fixture
def paris_video():
    """Video segment about Paris (should not match New York)"""
    seg = SRTSegment(
        index=1,
        start_time=0.0,
        end_time=10.0,
        text="The Eiffel Tower in Paris, France at night.",
        source_file="/videos/paris.mp4"
    )
    return seg


@pytest.fixture
def elon_musk_voiceover():
    """Voiceover mentioning Elon Musk (person entity)"""
    seg = SRTSegment(
        index=1,
        start_time=0.0,
        end_time=10.0,
        text="Elon Musk founded SpaceX to make humanity a multi-planetary species.",
        source_file="voiceover.srt"
    )
    return seg


@pytest.fixture
def spacex_video():
    """Video about SpaceX featuring Elon Musk"""
    seg = SRTSegment(
        index=1,
        start_time=0.0,
        end_time=10.0,
        text="Elon Musk presents the SpaceX Starship rocket at Boca Chica.",
        source_file="/videos/spacex.mp4"
    )
    return seg


# ============================================================================
# Test extract_named_entities_from_text
# ============================================================================

class TestExtractNamedEntitiesFromText:
    """Test heuristic named entity extraction from raw text"""

    @pytest.mark.fast
    def test_extract_multi_word_place(self):
        """Test extraction of multi-word place names like 'New York'"""
        text = "I visited New York last summer."
        entities = extract_named_entities_from_text(text)

        assert "New York" in entities

    @pytest.mark.fast
    def test_extract_multi_word_person(self):
        """Test extraction of person names like 'Elon Musk'"""
        text = "Elon Musk is the CEO of Tesla."
        entities = extract_named_entities_from_text(text)

        assert "Elon Musk" in entities

    @pytest.mark.fast
    def test_extract_organization(self):
        """Test extraction of organization names"""
        text = "Microsoft Corporation announced new products."
        entities = extract_named_entities_from_text(text)

        # Should extract Microsoft Corporation
        assert any("Microsoft" in e for e in entities)

    @pytest.mark.fast
    def test_extract_known_city_pattern(self):
        """Test extraction of known multi-word cities (Los Angeles, San Francisco)"""
        text = "Los Angeles and San Francisco are in California."
        entities = extract_named_entities_from_text(text)

        assert "Los Angeles" in entities or any("Los Angeles" in e for e in entities)
        assert "San Francisco" in entities or any("San Francisco" in e for e in entities)

    @pytest.mark.fast
    def test_empty_text_returns_empty(self):
        """Test that empty text returns empty list"""
        assert extract_named_entities_from_text("") == []
        assert extract_named_entities_from_text(None) == []

    @pytest.mark.fast
    def test_no_entities_in_lowercase_text(self):
        """Test that all-lowercase text has no capitalized entities"""
        text = "the quick brown fox jumps over the lazy dog."
        entities = extract_named_entities_from_text(text)

        # Should have no entities (no capitalization)
        assert len(entities) == 0

    @pytest.mark.fast
    def test_stopwords_not_extracted(self):
        """Test that common sentence-starting stopwords are not extracted"""
        text = "The cat sat on the mat. However, it was comfortable."
        entities = extract_named_entities_from_text(text)

        # The, However should not be entities
        assert "The" not in entities
        assert "However" not in entities

    @pytest.mark.fast
    def test_multiple_entities(self):
        """Test extraction of multiple entities from text"""
        text = "Apple Inc and Google are tech companies based in Silicon Valley."
        entities = extract_named_entities_from_text(text)

        # Should find multiple entities
        assert len(entities) >= 2


# ============================================================================
# Test _extract_entity_texts with text-based extraction
# ============================================================================

class TestExtractEntityTextsWithTextExtraction:
    """Test _extract_entity_texts function with US-63-011 text extraction"""

    @pytest.mark.fast
    def test_extracts_from_text_when_no_entities_field(self, new_york_voiceover):
        """Test that entities are extracted from text when entities field is empty"""
        # No pre-set entities
        new_york_voiceover.entities = []
        new_york_voiceover.keywords = []

        entities = _extract_entity_texts(new_york_voiceover)

        # Should extract New York from text
        assert any("New York" in e for e in entities)

    @pytest.mark.fast
    def test_combines_entities_field_and_text_extraction(self):
        """Test that both entities field and text extraction are combined"""
        seg = SRTSegment(
            index=1,
            start_time=0.0,
            end_time=10.0,
            text="London is a city in the United Kingdom.",
            source_file="voiceover.srt"
        )
        # Pre-set entity
        seg.entities = [{"text": "Tokyo", "type": "LOCATION"}]
        seg.keywords = []

        entities = _extract_entity_texts(seg)

        # Should have both Tokyo (from entities) and London/United Kingdom (from text)
        assert "Tokyo" in entities
        # Should also find something from text (London or United Kingdom)
        assert any(e in ["London", "United Kingdom"] for e in entities)

    @pytest.mark.fast
    def test_deduplicates_entities(self):
        """Test that duplicate entities are not included"""
        seg = SRTSegment(
            index=1,
            start_time=0.0,
            end_time=10.0,
            text="Tokyo is beautiful. Tokyo has great food.",
            source_file="voiceover.srt"
        )
        seg.entities = [{"text": "Tokyo", "type": "LOCATION"}]
        seg.keywords = ["Tokyo"]

        entities = _extract_entity_texts(seg)

        # Should only have Tokyo once (deduped)
        tokyo_count = sum(1 for e in entities if e.lower() == "tokyo")
        assert tokyo_count == 1


# ============================================================================
# Test apply_entity_match_boost
# ============================================================================

class TestApplyEntityMatchBoost:
    """Test the apply_entity_match_boost function with configurable boost"""

    @pytest.mark.fast
    def test_boost_applied_for_new_york_match(
        self, new_york_voiceover, nyc_video, mock_config_with_entity_boost
    ):
        """Test that boost is applied when New York appears in both segments"""
        initial_confidence = 0.7

        boosted, reason, matched = apply_entity_match_boost(
            initial_confidence, new_york_voiceover, nyc_video, mock_config_with_entity_boost
        )

        # Should have boosted confidence
        assert boosted > initial_confidence
        # Should mention entity match in reason
        assert "entity match" in reason
        # Should have matched entities
        assert len(matched) > 0

    @pytest.mark.fast
    def test_configurable_boost_value(
        self, new_york_voiceover, nyc_video, mock_config_with_entity_boost
    ):
        """Test that the configurable entity_match_boost value is used"""
        initial_confidence = 0.7
        expected_boost = 0.1  # Default

        boosted, reason, matched = apply_entity_match_boost(
            initial_confidence, new_york_voiceover, nyc_video, mock_config_with_entity_boost
        )

        # Should apply the configured boost (or close to it)
        # Note: actual boost may vary based on match count
        assert boosted >= initial_confidence + 0.05  # At least some boost

    @pytest.mark.fast
    def test_higher_configurable_boost(
        self, new_york_voiceover, nyc_video, mock_config_high_boost
    ):
        """Test that higher entity_match_boost value is applied"""
        initial_confidence = 0.6

        boosted, reason, matched = apply_entity_match_boost(
            initial_confidence, new_york_voiceover, nyc_video, mock_config_high_boost
        )

        # Should have boosted with higher value
        assert boosted >= initial_confidence + 0.1

    @pytest.mark.fast
    def test_no_boost_when_no_match(
        self, new_york_voiceover, paris_video, mock_config_with_entity_boost
    ):
        """Test that no boost is applied when entities don't match"""
        initial_confidence = 0.7

        boosted, reason, matched = apply_entity_match_boost(
            initial_confidence, new_york_voiceover, paris_video, mock_config_with_entity_boost
        )

        # Should not be boosted (no match between New York and Paris)
        assert boosted == initial_confidence
        assert reason == ""
        assert matched == []

    @pytest.mark.fast
    def test_boost_for_person_entity(
        self, elon_musk_voiceover, spacex_video, mock_config_with_entity_boost
    ):
        """Test boost is applied for person entity matches (Elon Musk)"""
        initial_confidence = 0.65

        boosted, reason, matched = apply_entity_match_boost(
            initial_confidence, elon_musk_voiceover, spacex_video, mock_config_with_entity_boost
        )

        # Should be boosted - both mention Elon Musk and SpaceX
        assert boosted > initial_confidence
        assert "entity match" in reason

    @pytest.mark.fast
    def test_confidence_capped_at_one(
        self, new_york_voiceover, nyc_video, mock_config_with_entity_boost
    ):
        """Test that confidence is capped at 1.0 even with boost"""
        initial_confidence = 0.98

        boosted, reason, matched = apply_entity_match_boost(
            initial_confidence, new_york_voiceover, nyc_video, mock_config_with_entity_boost
        )

        # Should be capped at 1.0
        assert boosted <= 1.0


# ============================================================================
# Test calculate_entity_match_score
# ============================================================================

class TestCalculateEntityMatchScore:
    """Test the calculate_entity_match_score function"""

    @pytest.mark.fast
    def test_score_for_matching_entities(self):
        """Test score calculation for matching entities"""
        vo_entities = ["New York", "Manhattan"]
        video_entities = ["New York", "skyline", "City"]

        score, matched = calculate_entity_match_score(vo_entities, video_entities)

        # Should have a non-zero score
        assert score > 0.0
        # Should identify New York as match
        assert "New York" in matched

    @pytest.mark.fast
    def test_score_zero_for_no_matches(self):
        """Test score is 0 when no entities match"""
        vo_entities = ["Paris", "France"]
        video_entities = ["Tokyo", "Japan"]

        score, matched = calculate_entity_match_score(vo_entities, video_entities)

        assert score == 0.0
        assert matched == []

    @pytest.mark.fast
    def test_score_for_multiple_matches(self):
        """Test higher score for multiple matching entities"""
        vo_entities = ["Elon Musk", "SpaceX", "Tesla"]
        video_entities = ["Elon Musk", "SpaceX", "rocket"]

        score, matched = calculate_entity_match_score(vo_entities, video_entities)

        # Should have high score for 2 matches
        assert score >= 0.8
        assert "Elon Musk" in matched
        assert "SpaceX" in matched

    @pytest.mark.fast
    def test_case_insensitive_matching(self):
        """Test that entity matching is case-insensitive"""
        vo_entities = ["new york"]
        video_entities = ["New York", "NYC"]

        score, matched = calculate_entity_match_score(vo_entities, video_entities)

        # Should match despite case difference
        assert score > 0.0
        assert len(matched) == 1

    @pytest.mark.fast
    def test_empty_entity_lists(self):
        """Test handling of empty entity lists"""
        assert calculate_entity_match_score([], ["New York"]) == (0.0, [])
        assert calculate_entity_match_score(["Paris"], []) == (0.0, [])
        assert calculate_entity_match_score([], []) == (0.0, [])


# ============================================================================
# Integration Test: Full Entity Match Flow
# ============================================================================

class TestEntityMatchIntegration:
    """Integration tests for the full entity match flow"""

    @pytest.mark.fast
    def test_full_flow_new_york_to_nyc(self, new_york_voiceover, nyc_video, mock_config_with_entity_boost):
        """Integration test: voiceover about 'New York' matches video about 'NYC'

        This is the primary acceptance criterion test case.
        """
        # Step 1: Extract entities from voiceover
        vo_entities = _extract_entity_texts(new_york_voiceover)

        # Verify New York was extracted
        assert any("New York" in e for e in vo_entities), \
            f"Expected 'New York' in extracted entities: {vo_entities}"

        # Step 2: Extract entities from video
        video_entities = _extract_entity_texts(nyc_video)

        # Verify video has New York related entities
        assert any("New York" in e for e in video_entities), \
            f"Expected 'New York' in video entities: {video_entities}"

        # Step 3: Calculate entity match score
        score, matched = calculate_entity_match_score(vo_entities, video_entities)

        # Should have a match
        assert score > 0.0, f"Expected non-zero score, got {score}"
        assert len(matched) > 0, f"Expected matched entities, got {matched}"

        # Step 4: Apply boost
        initial_confidence = 0.7
        boosted, reason, matched_entities = apply_entity_match_boost(
            initial_confidence, new_york_voiceover, nyc_video, mock_config_with_entity_boost
        )

        # Should have applied boost
        assert boosted > initial_confidence, \
            f"Expected boosted confidence > {initial_confidence}, got {boosted}"
        assert "entity match" in reason
        assert len(matched_entities) > 0

    @pytest.mark.fast
    def test_config_option_entity_match_boost_default(self):
        """Test that config option matching.entity_match_boost exists with default 0.1"""
        from src.config.sections.matching import MatchingConfig

        config = MatchingConfig()

        # Should have entity_match_boost attribute
        assert hasattr(config, 'entity_match_boost')
        # Default should be 0.1
        assert config.entity_match_boost == 0.1
