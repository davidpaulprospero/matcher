"""
Tests for LLM reasoning quality validation.

US-005: Add LLM reasoning quality validation
- validate_llm_reasoning() checks if LLM reasoning mentions specific keywords
- Flags low-quality reasoning that is generic (< 3 specific references)
- Logs warning when LLM provides generic reasoning
"""

import logging
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.llm_providers import (
    validate_llm_reasoning,
    ReasoningValidation,
    GENERIC_REASONING_PHRASES,
    _extract_keywords,
)


# Mark all tests as unit tests
pytestmark = pytest.mark.unit


class TestValidateLLMReasoningFunction:
    """Test validate_llm_reasoning() function exists and has correct signature."""

    @pytest.mark.fast
    def test_function_exists(self):
        """validate_llm_reasoning function exists in module."""
        from src.matching import llm_providers
        assert hasattr(llm_providers, 'validate_llm_reasoning')

    @pytest.mark.fast
    def test_function_callable(self):
        """validate_llm_reasoning is callable."""
        assert callable(validate_llm_reasoning)

    @pytest.mark.fast
    def test_function_accepts_required_args(self):
        """Function accepts reasoning and voiceover_text args."""
        result = validate_llm_reasoning(
            reasoning="test",
            voiceover_text="test"
        )
        assert isinstance(result, ReasoningValidation)

    @pytest.mark.fast
    def test_function_accepts_optional_video_text(self):
        """Function accepts optional video_text argument."""
        result = validate_llm_reasoning(
            reasoning="test",
            voiceover_text="test",
            video_text="video content"
        )
        assert isinstance(result, ReasoningValidation)

    @pytest.mark.fast
    def test_function_accepts_min_specific_references(self):
        """Function accepts min_specific_references argument."""
        result = validate_llm_reasoning(
            reasoning="test",
            voiceover_text="test",
            min_specific_references=5
        )
        assert isinstance(result, ReasoningValidation)


class TestReasoningValidationDataclass:
    """Test ReasoningValidation dataclass structure."""

    @pytest.mark.fast
    def test_dataclass_exists(self):
        """ReasoningValidation dataclass exists."""
        assert ReasoningValidation is not None

    @pytest.mark.fast
    def test_has_is_valid_field(self):
        """ReasoningValidation has is_valid field."""
        result = ReasoningValidation(is_valid=True, specific_references=0, matched_keywords=[])
        assert hasattr(result, 'is_valid')
        assert result.is_valid is True

    @pytest.mark.fast
    def test_has_specific_references_field(self):
        """ReasoningValidation has specific_references field."""
        result = ReasoningValidation(is_valid=True, specific_references=5, matched_keywords=[])
        assert hasattr(result, 'specific_references')
        assert result.specific_references == 5

    @pytest.mark.fast
    def test_has_matched_keywords_field(self):
        """ReasoningValidation has matched_keywords field."""
        result = ReasoningValidation(is_valid=True, specific_references=2, matched_keywords=['word1', 'word2'])
        assert hasattr(result, 'matched_keywords')
        assert result.matched_keywords == ['word1', 'word2']

    @pytest.mark.fast
    def test_has_warning_message_field(self):
        """ReasoningValidation has optional warning_message field."""
        result = ReasoningValidation(
            is_valid=False,
            specific_references=0,
            matched_keywords=[],
            warning_message="Warning text"
        )
        assert hasattr(result, 'warning_message')
        assert result.warning_message == "Warning text"

    @pytest.mark.fast
    def test_warning_message_defaults_to_none(self):
        """warning_message defaults to None."""
        result = ReasoningValidation(is_valid=True, specific_references=0, matched_keywords=[])
        assert result.warning_message is None


class TestGenericReasoningDetection:
    """Test detection of generic/low-quality reasoning."""

    @pytest.mark.fast
    def test_short_good_match_flagged(self):
        """Short 'good match' reasoning is flagged as generic."""
        result = validate_llm_reasoning(
            reasoning="good match",
            voiceover_text="The earthquake destroyed buildings downtown"
        )
        assert result.is_valid is False
        assert "Generic" in result.warning_message

    @pytest.mark.fast
    def test_short_topic_match_flagged(self):
        """Short 'topic match' reasoning is flagged as generic."""
        result = validate_llm_reasoning(
            reasoning="topic match",
            voiceover_text="Scientists discovered new species"
        )
        assert result.is_valid is False
        assert "Generic" in result.warning_message

    @pytest.mark.fast
    def test_short_similar_flagged(self):
        """Short 'similar' reasoning is flagged as generic."""
        result = validate_llm_reasoning(
            reasoning="similar content",
            voiceover_text="The weather forecast shows rain"
        )
        assert result.is_valid is False

    @pytest.mark.fast
    def test_short_relevant_flagged(self):
        """Short 'relevant' reasoning is flagged as generic."""
        result = validate_llm_reasoning(
            reasoning="relevant",
            voiceover_text="Stock market prices fluctuated today"
        )
        assert result.is_valid is False

    @pytest.mark.fast
    def test_longer_reasoning_with_generic_phrase_not_immediately_flagged(self):
        """Longer reasoning with generic phrase is checked for keywords."""
        # This is long enough that it's not immediately flagged as generic
        # Note: keywords must match exactly (earthquake != earthquakes)
        result = validate_llm_reasoning(
            reasoning="This video about earthquake and downtown buildings with damage is a good match",
            voiceover_text="The earthquake damaged buildings downtown"
        )
        # Should find keywords: earthquake, downtown, buildings, damage
        assert result.specific_references >= 3


class TestSpecificReasoningValidation:
    """Test validation of specific/high-quality reasoning."""

    @pytest.mark.fast
    def test_specific_reasoning_valid(self):
        """Reasoning with specific keywords from voiceover is valid."""
        result = validate_llm_reasoning(
            reasoning="Shows earthquake damage in downtown area with collapsed buildings and rescue operations",
            voiceover_text="The earthquake destroyed buildings downtown causing widespread damage"
        )
        assert result.is_valid is True
        assert result.specific_references >= 3

    @pytest.mark.fast
    def test_matched_keywords_populated(self):
        """matched_keywords contains the actual matched words."""
        result = validate_llm_reasoning(
            reasoning="Video shows hurricane Florence approaching coastline with evacuation",
            voiceover_text="Hurricane Florence caused massive evacuation of the coastline"
        )
        assert len(result.matched_keywords) > 0
        # Common keywords should be matched
        assert any(kw in ['hurricane', 'florence', 'evacuation', 'coastline'] for kw in result.matched_keywords)

    @pytest.mark.fast
    def test_video_text_keywords_also_count(self):
        """Keywords from video_text also count as specific references."""
        result = validate_llm_reasoning(
            reasoning="Mentions volcano eruption and lava flow with rescue helicopters",
            voiceover_text="Simple narration text",
            video_text="Volcano eruption with lava flow and rescue operations by helicopter"
        )
        # Should find: volcano, eruption, lava, rescue, helicopter
        assert result.specific_references >= 3

    @pytest.mark.fast
    def test_combined_vo_and_video_keywords(self):
        """Keywords from both voiceover and video are considered."""
        result = validate_llm_reasoning(
            reasoning="Ocean waves and surfing competition with athlete performance",
            voiceover_text="Championship surfing competition",
            video_text="Ocean waves and athletic performance"
        )
        # Keywords from both sources should be matched
        assert result.specific_references >= 3


class TestWarningLogging:
    """Test warning logging for low-quality reasoning."""

    @pytest.mark.fast
    def test_generic_reasoning_logs_warning(self, caplog):
        """Generic reasoning logs a warning."""
        with caplog.at_level(logging.WARNING):
            validate_llm_reasoning(
                reasoning="good match",
                voiceover_text="Test voiceover text"
            )
        assert any("Generic" in record.message for record in caplog.records)

    @pytest.mark.fast
    def test_low_quality_reasoning_logs_warning(self, caplog):
        """Low-quality reasoning with few references logs warning."""
        with caplog.at_level(logging.WARNING):
            validate_llm_reasoning(
                reasoning="Something about video",
                voiceover_text="Completely different topic here"
            )
        assert any("Low-quality" in record.message or "Generic" in record.message for record in caplog.records)

    @pytest.mark.fast
    def test_empty_reasoning_logs_warning(self, caplog):
        """Empty reasoning logs a warning."""
        with caplog.at_level(logging.WARNING):
            validate_llm_reasoning(
                reasoning="",
                voiceover_text="Test voiceover text"
            )
        assert any("Empty reasoning" in record.message for record in caplog.records)

    @pytest.mark.fast
    def test_valid_reasoning_no_warning(self, caplog):
        """Valid specific reasoning does not log warning."""
        with caplog.at_level(logging.WARNING):
            validate_llm_reasoning(
                reasoning="Video shows earthquake damage in downtown area with collapsed buildings",
                voiceover_text="The earthquake destroyed buildings downtown"
            )
        # Should not have any warning about low-quality
        assert not any("Low-quality" in record.message for record in caplog.records)
        assert not any("Generic" in record.message for record in caplog.records)


class TestMinSpecificReferencesThreshold:
    """Test min_specific_references threshold parameter."""

    @pytest.mark.fast
    def test_default_threshold_is_3(self):
        """Default threshold requires 3 specific references."""
        # Only 2 matching keywords
        result = validate_llm_reasoning(
            reasoning="Shows earthquake damage",
            voiceover_text="The earthquake caused damage"
        )
        assert result.is_valid is False  # Only 2: earthquake, damage

    @pytest.mark.fast
    def test_custom_threshold_1(self):
        """Custom threshold of 1 allows single reference."""
        result = validate_llm_reasoning(
            reasoning="Shows earthquake",
            voiceover_text="The earthquake happened",
            min_specific_references=1
        )
        assert result.is_valid is True
        assert result.specific_references >= 1

    @pytest.mark.fast
    def test_custom_threshold_5(self):
        """Custom threshold of 5 requires 5 references."""
        result = validate_llm_reasoning(
            reasoning="Shows earthquake and damage",
            voiceover_text="The earthquake caused damage",
            min_specific_references=5
        )
        assert result.is_valid is False
        assert result.specific_references < 5


class TestExtractKeywordsHelper:
    """Test _extract_keywords helper function."""

    @pytest.mark.fast
    def test_basic_extraction(self):
        """Extracts basic keywords from text."""
        keywords = _extract_keywords("The earthquake destroyed buildings")
        assert 'earthquake' in keywords
        assert 'destroyed' in keywords
        assert 'buildings' in keywords

    @pytest.mark.fast
    def test_filters_short_words(self):
        """Filters words shorter than min_length."""
        keywords = _extract_keywords("I am at the zoo")
        assert 'zoo' in keywords  # 3 chars, included
        assert 'am' not in keywords  # 2 chars, excluded
        assert 'at' not in keywords  # 2 chars, excluded

    @pytest.mark.fast
    def test_filters_stopwords(self):
        """Filters common stopwords."""
        keywords = _extract_keywords("The building was destroyed with fire")
        assert 'the' not in keywords
        assert 'was' not in keywords
        assert 'with' not in keywords
        assert 'building' in keywords
        assert 'destroyed' in keywords
        assert 'fire' in keywords

    @pytest.mark.fast
    def test_handles_empty_text(self):
        """Handles empty text gracefully."""
        keywords = _extract_keywords("")
        assert keywords == set()

    @pytest.mark.fast
    def test_handles_none_text(self):
        """Handles None text gracefully."""
        keywords = _extract_keywords(None)
        assert keywords == set()

    @pytest.mark.fast
    def test_lowercase_normalization(self):
        """Keywords are normalized to lowercase."""
        keywords = _extract_keywords("The EARTHQUAKE Destroyed BUILDINGS")
        assert 'earthquake' in keywords
        assert 'EARTHQUAKE' not in keywords


class TestGenericPhrasesConstant:
    """Test GENERIC_REASONING_PHRASES constant."""

    @pytest.mark.fast
    def test_constant_exists(self):
        """GENERIC_REASONING_PHRASES constant exists."""
        assert GENERIC_REASONING_PHRASES is not None

    @pytest.mark.fast
    def test_contains_common_generic_phrases(self):
        """Contains common generic matching phrases."""
        assert "good match" in GENERIC_REASONING_PHRASES
        assert "topic match" in GENERIC_REASONING_PHRASES
        assert "relevant" in GENERIC_REASONING_PHRASES

    @pytest.mark.fast
    def test_is_frozenset(self):
        """GENERIC_REASONING_PHRASES is a frozenset (immutable)."""
        assert isinstance(GENERIC_REASONING_PHRASES, frozenset)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.fast
    def test_none_reasoning(self):
        """Handles None reasoning gracefully."""
        result = validate_llm_reasoning(
            reasoning=None,
            voiceover_text="Test voiceover"
        )
        assert result.is_valid is False
        assert result.warning_message is not None

    @pytest.mark.fast
    def test_whitespace_only_reasoning(self):
        """Handles whitespace-only reasoning."""
        result = validate_llm_reasoning(
            reasoning="   ",
            voiceover_text="Test voiceover"
        )
        # Should be treated as invalid (no keywords)
        assert result.is_valid is False

    @pytest.mark.fast
    def test_none_voiceover_text(self):
        """Handles None voiceover_text gracefully."""
        result = validate_llm_reasoning(
            reasoning="Some reasoning about earthquakes",
            voiceover_text=None
        )
        # Should work but may not find matches
        assert isinstance(result, ReasoningValidation)

    @pytest.mark.fast
    def test_special_characters_in_reasoning(self):
        """Handles special characters in reasoning."""
        result = validate_llm_reasoning(
            reasoning="Shows earthquake!!! & damage??? @downtown",
            voiceover_text="The earthquake caused damage downtown"
        )
        assert isinstance(result, ReasoningValidation)
        # Should still extract: earthquake, damage, downtown
        assert result.specific_references >= 3

    @pytest.mark.fast
    def test_unicode_characters(self):
        """Handles unicode characters in text."""
        result = validate_llm_reasoning(
            reasoning="Shows café and résumé content",
            voiceover_text="Discussion about café culture"
        )
        # Should handle gracefully (only extract ASCII words)
        assert isinstance(result, ReasoningValidation)

    @pytest.mark.fast
    def test_very_long_reasoning(self):
        """Handles very long reasoning strings."""
        # Include enough matching keywords
        long_reasoning = "earthquake damage buildings rescue teams downtown " * 100
        result = validate_llm_reasoning(
            reasoning=long_reasoning,
            voiceover_text="The earthquake caused damage to buildings downtown with rescue teams responding"
        )
        assert isinstance(result, ReasoningValidation)
        # Should find: earthquake, damage, buildings, rescue, teams, downtown
        assert result.is_valid is True


class TestRealWorldExamples:
    """Test with realistic voiceover and reasoning examples."""

    @pytest.mark.fast
    def test_documentary_earthquake_good_reasoning(self):
        """Good reasoning for documentary about earthquake."""
        result = validate_llm_reasoning(
            reasoning="Video shows collapsed buildings in downtown area after the earthquake with rescue teams searching through rubble",
            voiceover_text="The devastating earthquake left buildings collapsed downtown as rescue teams worked through the rubble"
        )
        assert result.is_valid is True
        assert result.specific_references >= 3

    @pytest.mark.fast
    def test_documentary_earthquake_bad_reasoning(self):
        """Generic reasoning for documentary should be flagged."""
        result = validate_llm_reasoning(
            reasoning="good match",
            voiceover_text="The devastating earthquake left buildings collapsed downtown as rescue teams worked through the rubble"
        )
        assert result.is_valid is False

    @pytest.mark.fast
    def test_cooking_show_specific_reasoning(self):
        """Specific reasoning for cooking show."""
        result = validate_llm_reasoning(
            reasoning="Chef preparing pasta with tomato sauce and fresh basil in the kitchen",
            voiceover_text="Watch as the chef prepares homemade pasta with fresh tomato sauce and basil"
        )
        assert result.is_valid is True

    @pytest.mark.fast
    def test_nature_documentary_specific_reasoning(self):
        """Specific reasoning for nature documentary."""
        result = validate_llm_reasoning(
            reasoning="Lions hunting zebras on the African savanna during migration",
            voiceover_text="During the annual migration, lions prey on zebras crossing the African savanna",
            video_text="Safari footage of lion pride hunting behavior"
        )
        assert result.is_valid is True
        # Should find: lions, hunting, zebras, african, savanna, migration
        assert result.specific_references >= 3
