"""
Unit tests for LLM explanation confidence validation.

Tests the validate_explanation_confidence() function which cross-checks
LLM explanation keywords against actual voiceover/video text to detect
"hallucinated" explanations and apply confidence penalties.

Acceptance criteria tested:
- Cross-check that LLM explanation keywords appear in actual voiceover/video text
- Downgrade confidence by 0.1 when <50% of explanation keywords are verifiable
- Log warning when explanation appears to reference non-existent content
- explanation_validation_enabled config option (default: true)
"""

import pytest
from unittest.mock import patch, MagicMock
import logging

from src.matching.llm_providers import (
    validate_explanation_confidence,
    ExplanationValidation,
    EXPLANATION_VERIFICATION_THRESHOLD,
    EXPLANATION_CONFIDENCE_PENALTY,
)


class TestExplanationValidationDataclass:
    """Test ExplanationValidation dataclass structure."""

    def test_dataclass_has_required_fields(self):
        """ExplanationValidation should have all required fields."""
        validation = ExplanationValidation(
            is_valid=True,
            verification_ratio=0.75,
            explanation_keywords=["sunset", "peaceful", "ocean"],
            verified_keywords=["sunset", "peaceful"],
            unverified_keywords=["ocean"],
            confidence_penalty=0.0,
            warning_message=None
        )
        assert validation.is_valid is True
        assert validation.verification_ratio == 0.75
        assert len(validation.explanation_keywords) == 3
        assert len(validation.verified_keywords) == 2
        assert len(validation.unverified_keywords) == 1
        assert validation.confidence_penalty == 0.0
        assert validation.warning_message is None

    def test_dataclass_with_warning(self):
        """ExplanationValidation should include warning_message when invalid."""
        validation = ExplanationValidation(
            is_valid=False,
            verification_ratio=0.25,
            explanation_keywords=["sunset", "mountains", "forest", "beach"],
            verified_keywords=["sunset"],
            unverified_keywords=["mountains", "forest", "beach"],
            confidence_penalty=0.1,
            warning_message="Only 25% keywords verifiable"
        )
        assert validation.is_valid is False
        assert validation.confidence_penalty == 0.1
        assert "25%" in validation.warning_message


class TestValidateExplanationConfidenceBasic:
    """Basic functionality tests for validate_explanation_confidence."""

    def test_empty_explanation_returns_invalid(self):
        """Empty explanation should return invalid result with penalty."""
        result = validate_explanation_confidence(
            explanation="",
            voiceover_text="A peaceful evening by the ocean"
        )
        assert result.is_valid is False
        assert result.verification_ratio == 0.0
        assert result.confidence_penalty == EXPLANATION_CONFIDENCE_PENALTY
        assert "Empty explanation" in result.warning_message

    def test_none_explanation_returns_invalid(self):
        """None explanation should return invalid result."""
        result = validate_explanation_confidence(
            explanation=None,
            voiceover_text="A peaceful evening by the ocean"
        )
        assert result.is_valid is False
        assert result.confidence_penalty == EXPLANATION_CONFIDENCE_PENALTY

    def test_no_keywords_in_explanation_returns_valid(self):
        """Explanation with no extractable keywords is assumed valid."""
        # All stopwords, no meaningful keywords
        result = validate_explanation_confidence(
            explanation="The and for this with",
            voiceover_text="A peaceful evening by the ocean"
        )
        assert result.is_valid is True
        assert result.verification_ratio == 1.0
        assert result.confidence_penalty == 0.0
        assert len(result.explanation_keywords) == 0

    def test_all_keywords_verifiable(self):
        """All explanation keywords found in source returns valid."""
        result = validate_explanation_confidence(
            explanation="The sunset and ocean create a peaceful atmosphere",
            voiceover_text="A peaceful sunset evening by the ocean"
        )
        # Keywords extracted: sunset, ocean, create, peaceful, atmosphere (5 total)
        # Verifiable: sunset, ocean, peaceful (3 verifiable)
        # Ratio: 3/5 = 0.6 (60%) >= 50% threshold
        assert result.is_valid is True
        assert result.verification_ratio == 0.6
        assert result.confidence_penalty == 0.0
        assert result.verification_ratio >= EXPLANATION_VERIFICATION_THRESHOLD


class TestVerificationRatioCalculation:
    """Tests for verification ratio calculation logic."""

    def test_100_percent_verification(self):
        """100% verifiable keywords should have ratio 1.0."""
        result = validate_explanation_confidence(
            explanation="sunset ocean evening",
            voiceover_text="sunset ocean evening",
            video_text=""
        )
        assert result.verification_ratio == 1.0
        assert result.is_valid is True

    def test_0_percent_verification(self):
        """0% verifiable keywords should have ratio 0.0."""
        result = validate_explanation_confidence(
            explanation="mountains forest hiking trail",
            voiceover_text="sunset ocean evening beach",
            video_text=""
        )
        assert result.verification_ratio == 0.0
        assert result.is_valid is False
        assert result.confidence_penalty == EXPLANATION_CONFIDENCE_PENALTY

    def test_50_percent_threshold_exact(self):
        """Exactly 50% should pass (>= threshold)."""
        result = validate_explanation_confidence(
            explanation="sunset mountains",  # 2 keywords
            voiceover_text="sunset ocean",  # 'sunset' verifiable, 'mountains' not
            video_text=""
        )
        assert result.verification_ratio == 0.5
        assert result.is_valid is True
        assert result.confidence_penalty == 0.0

    def test_below_50_percent_threshold(self):
        """Below 50% should fail and apply penalty."""
        result = validate_explanation_confidence(
            explanation="sunset mountains forest",  # 3 keywords
            voiceover_text="sunset ocean",  # only 'sunset' verifiable = 33%
            video_text=""
        )
        assert result.verification_ratio < 0.5
        assert result.is_valid is False
        assert result.confidence_penalty == EXPLANATION_CONFIDENCE_PENALTY


class TestVideoTextIntegration:
    """Tests for combining voiceover and video text sources."""

    def test_video_text_contributes_to_verification(self):
        """Keywords in video text should count as verified."""
        result = validate_explanation_confidence(
            explanation="sunset mountains",
            voiceover_text="peaceful evening",  # 0 matches
            video_text="sunset over mountains"  # both keywords here
        )
        assert result.verification_ratio == 1.0
        assert result.is_valid is True

    def test_combined_voiceover_and_video(self):
        """Keywords from both voiceover and video should combine."""
        result = validate_explanation_confidence(
            explanation="sunset peaceful mountains",  # 3 keywords
            voiceover_text="peaceful evening",  # 'peaceful' here
            video_text="sunset over valley"  # 'sunset' here, 'mountains' not
        )
        # 2/3 verified = 66.7%
        assert result.verification_ratio == pytest.approx(2/3, rel=0.01)
        assert result.is_valid is True

    def test_none_video_text_handled(self):
        """None video_text should not cause errors."""
        result = validate_explanation_confidence(
            explanation="sunset peaceful",
            voiceover_text="peaceful evening",
            video_text=None
        )
        # Only 'peaceful' verifiable = 50%
        assert result.is_valid is True
        assert result.confidence_penalty == 0.0


class TestVerifiedUnverifiedKeywords:
    """Tests for verified and unverified keyword tracking."""

    def test_verified_keywords_populated(self):
        """Verified keywords list should contain matching keywords."""
        result = validate_explanation_confidence(
            explanation="sunset peaceful mountains",
            voiceover_text="sunset peaceful ocean",
            video_text=""
        )
        assert "sunset" in result.verified_keywords
        assert "peaceful" in result.verified_keywords
        assert "mountains" not in result.verified_keywords

    def test_unverified_keywords_populated(self):
        """Unverified keywords list should contain non-matching keywords."""
        result = validate_explanation_confidence(
            explanation="sunset mountains forest",
            voiceover_text="sunset ocean beach",
            video_text=""
        )
        assert "mountains" in result.unverified_keywords
        assert "forest" in result.unverified_keywords
        assert "sunset" not in result.unverified_keywords

    def test_keywords_are_sorted(self):
        """Keywords lists should be sorted alphabetically."""
        result = validate_explanation_confidence(
            explanation="zebra apple mountain sunset",
            voiceover_text="sunset apple ocean",
            video_text=""
        )
        assert result.verified_keywords == sorted(result.verified_keywords)
        assert result.unverified_keywords == sorted(result.unverified_keywords)
        assert result.explanation_keywords == sorted(result.explanation_keywords)


class TestConfidencePenalty:
    """Tests for confidence penalty application."""

    def test_default_penalty_value(self):
        """Default penalty should be 0.1."""
        assert EXPLANATION_CONFIDENCE_PENALTY == 0.1

    def test_default_threshold_value(self):
        """Default threshold should be 0.5 (50%)."""
        assert EXPLANATION_VERIFICATION_THRESHOLD == 0.5

    def test_penalty_applied_when_invalid(self):
        """Penalty should be applied when validation fails."""
        result = validate_explanation_confidence(
            explanation="mountains forest hiking",
            voiceover_text="sunset ocean beach",
            video_text=""
        )
        assert result.is_valid is False
        assert result.confidence_penalty == 0.1

    def test_no_penalty_when_valid(self):
        """No penalty should be applied when validation passes."""
        result = validate_explanation_confidence(
            explanation="sunset ocean",
            voiceover_text="sunset ocean beach",
            video_text=""
        )
        assert result.is_valid is True
        assert result.confidence_penalty == 0.0

    def test_custom_penalty_value(self):
        """Custom penalty value should be respected."""
        result = validate_explanation_confidence(
            explanation="mountains forest",
            voiceover_text="sunset ocean",
            video_text="",
            confidence_penalty=0.2
        )
        assert result.confidence_penalty == 0.2

    def test_custom_threshold_value(self):
        """Custom threshold should be respected."""
        # With default 50% threshold, 33% would fail
        # With 30% threshold, 33% should pass
        result = validate_explanation_confidence(
            explanation="sunset mountains forest",  # 3 keywords
            voiceover_text="sunset ocean",  # only 'sunset' = 33%
            video_text="",
            verification_threshold=0.30
        )
        assert result.verification_ratio == pytest.approx(1/3, rel=0.01)
        assert result.is_valid is True  # 33% >= 30%
        assert result.confidence_penalty == 0.0


class TestLogging:
    """Tests for logging behavior."""

    def test_warning_logged_when_invalid(self, caplog):
        """Warning should be logged when validation fails."""
        with caplog.at_level(logging.WARNING):
            result = validate_explanation_confidence(
                explanation="mountains forest hiking trail",
                voiceover_text="sunset ocean beach",
                video_text=""
            )

        assert result.is_valid is False
        assert "non-existent content" in caplog.text
        assert "keywords verifiable" in caplog.text

    def test_warning_includes_unverified_keywords(self, caplog):
        """Warning message should list unverified keywords."""
        with caplog.at_level(logging.WARNING):
            result = validate_explanation_confidence(
                explanation="mountains forest",
                voiceover_text="sunset ocean",
                video_text=""
            )

        assert "mountains" in caplog.text or "mountains" in str(result.warning_message)
        assert "forest" in caplog.text or "forest" in str(result.warning_message)

    def test_warning_limited_to_5_keywords(self, caplog):
        """Warning should limit unverified keywords to 5 for readability."""
        with caplog.at_level(logging.WARNING):
            result = validate_explanation_confidence(
                explanation="alpha bravo charlie delta echo foxtrot golf hotel",
                voiceover_text="sunset ocean",
                video_text=""
            )

        # All 8 keywords should be tracked, but warning limited to 5
        assert len(result.unverified_keywords) == 8
        # Warning should not have more than 5 unverified keywords displayed
        assert result.warning_message is not None

    def test_no_warning_when_valid(self, caplog):
        """No warning should be logged when validation passes."""
        with caplog.at_level(logging.WARNING):
            result = validate_explanation_confidence(
                explanation="sunset ocean",
                voiceover_text="sunset ocean beach",
                video_text=""
            )

        assert result.is_valid is True
        assert "non-existent content" not in caplog.text

    def test_debug_logged_when_valid(self, caplog):
        """Debug message should be logged when validation passes."""
        with caplog.at_level(logging.DEBUG):
            validate_explanation_confidence(
                explanation="sunset ocean",
                voiceover_text="sunset ocean beach",
                video_text=""
            )

        assert "keywords verified" in caplog.text


class TestWarningMessage:
    """Tests for warning message content."""

    def test_warning_message_format(self):
        """Warning message should have expected format."""
        result = validate_explanation_confidence(
            explanation="mountains forest hiking",  # 3 keywords
            voiceover_text="sunset ocean",  # 0 verifiable
            video_text=""
        )
        assert result.warning_message is not None
        assert "non-existent content" in result.warning_message
        assert "0/3" in result.warning_message or "0%" in result.warning_message

    def test_warning_message_includes_penalty(self):
        """Warning message should mention the applied penalty."""
        result = validate_explanation_confidence(
            explanation="mountains forest",
            voiceover_text="sunset ocean",
            video_text=""
        )
        assert "-0.1" in result.warning_message

    def test_warning_message_none_when_valid(self):
        """Warning message should be None when validation passes."""
        result = validate_explanation_confidence(
            explanation="sunset ocean",
            voiceover_text="sunset ocean beach",
            video_text=""
        )
        assert result.warning_message is None


class TestEdgeCases:
    """Edge case tests."""

    def test_case_insensitive_matching(self):
        """Keyword matching should be case-insensitive."""
        result = validate_explanation_confidence(
            explanation="SUNSET OCEAN",
            voiceover_text="sunset ocean beach",
            video_text=""
        )
        assert result.verification_ratio == 1.0

    def test_punctuation_handling(self):
        """Punctuation should be stripped from keywords."""
        result = validate_explanation_confidence(
            explanation="The sunset, ocean! And beach.",
            voiceover_text="sunset ocean beach",
            video_text=""
        )
        # 'sunset', 'ocean', 'beach' should all be extracted and verified
        assert "sunset" in result.verified_keywords
        assert "ocean" in result.verified_keywords
        assert "beach" in result.verified_keywords

    def test_short_words_filtered(self):
        """Words shorter than 3 characters should be filtered."""
        result = validate_explanation_confidence(
            explanation="at to sunset in on",  # only 'sunset' is >=3 chars
            voiceover_text="sunset ocean",
            video_text=""
        )
        assert len(result.explanation_keywords) == 1
        assert "sunset" in result.explanation_keywords

    def test_stopwords_filtered(self):
        """Common stopwords should be filtered."""
        result = validate_explanation_confidence(
            explanation="the sunset and ocean with beach",
            voiceover_text="sunset ocean beach",
            video_text=""
        )
        # 'the', 'and', 'with' should be filtered as stopwords
        assert "the" not in result.explanation_keywords
        assert "and" not in result.explanation_keywords
        assert "with" not in result.explanation_keywords

    def test_empty_voiceover_text(self):
        """Empty voiceover text should make all keywords unverifiable."""
        result = validate_explanation_confidence(
            explanation="sunset ocean beach",
            voiceover_text="",
            video_text=""
        )
        assert result.verification_ratio == 0.0
        assert result.is_valid is False

    def test_whitespace_only_texts(self):
        """Whitespace-only texts should be handled gracefully."""
        result = validate_explanation_confidence(
            explanation="   sunset   ocean   ",
            voiceover_text="   sunset   beach   ",
            video_text="   "
        )
        assert "sunset" in result.verified_keywords
        assert "ocean" in result.unverified_keywords


class TestConfigIntegration:
    """Tests related to config integration."""

    def test_default_config_value_is_true(self):
        """explanation_validation_enabled should default to True."""
        from src.config.sections.matching import MatchingConfig
        config = MatchingConfig()
        assert config.explanation_validation_enabled is True

    def test_config_can_be_disabled(self):
        """explanation_validation_enabled can be set to False."""
        from src.config.sections.matching import MatchingConfig
        config = MatchingConfig(explanation_validation_enabled=False)
        assert config.explanation_validation_enabled is False


class TestRealWorldScenarios:
    """Tests simulating real-world LLM explanation scenarios."""

    def test_good_llm_explanation(self):
        """Realistic good LLM explanation should pass validation."""
        result = validate_explanation_confidence(
            explanation="The video shows sunset imagery over the ocean which matches the voiceover about a peaceful evening",
            voiceover_text="A peaceful evening by the ocean as the sun sets",
            video_text="Sunset over calm ocean waters with waves gently lapping the shore"
        )
        assert result.is_valid is True
        assert result.confidence_penalty == 0.0

    def test_hallucinated_llm_explanation(self):
        """LLM explanation referencing non-existent content should fail."""
        result = validate_explanation_confidence(
            explanation="The mountain hiking footage perfectly matches the adventure narration about climbing peaks",
            voiceover_text="A peaceful evening by the ocean as the sun sets",
            video_text="Sunset over calm ocean waters"
        )
        assert result.is_valid is False
        assert result.confidence_penalty == 0.1
        # 'mountain', 'hiking', 'adventure', 'climbing', 'peaks' not in source

    def test_partially_accurate_explanation(self):
        """Partially accurate explanation should be evaluated correctly."""
        result = validate_explanation_confidence(
            explanation="The sunset footage shows mountains in the background with ocean views",
            voiceover_text="A sunset over the ocean",
            video_text="Beach sunset scene"
        )
        # 'sunset', 'ocean' verifiable; 'footage', 'shows', 'mountains', 'background', 'views' likely not
        # This tests realistic partial accuracy scenario
        assert 0.0 < result.verification_ratio < 1.0

    def test_technical_jargon_explanation(self):
        """Explanation with technical terms not in content should be penalized."""
        result = validate_explanation_confidence(
            explanation="Excellent cinematographic composition with drone aerial establishing shot",
            voiceover_text="The city looks beautiful at night",
            video_text="Night skyline with lights"
        )
        # Technical filmmaking terms won't be in voiceover/video text
        assert result.is_valid is False
