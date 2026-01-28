"""
Tests for multilingual caption fetching (US-003 Sprint 8).

Tests language fallback chains, Unicode handling, mixed-language videos,
and language code normalization for caption-first mode.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from src.caption_fetcher import (
    CaptionFetcher,
    CaptionResult,
    CaptionSegment,
    AvailableLanguage,
    is_valid_language_code,
    validate_language_config,
)


class TestMultilingualCaptions:
    """Test suite for multilingual caption fetching scenarios (US-003 Sprint 8).

    Tests cover:
    - Fallback chain behavior when preferred language unavailable
    - Unicode handling for various character sets
    - Mixed-language video selection
    - Language code case normalization
    - Empty fallback_languages defaults
    """

    @pytest.mark.fast
    def test_fallback_chain_preferred_unavailable_2nd_fallback_succeeds(self):
        """Test fallback chain: preferred (en) unavailable, skip es, de succeeds.

        Scenario: en -> es -> de chain where only de is available.
        This verifies the chain properly iterates through fallbacks.
        """
        fetcher = CaptionFetcher()

        # Only German (de) is available, en and es are not
        available = [
            AvailableLanguage("de", "German", False),
            AvailableLanguage("fr", "French", False),
        ]

        # Preferred is 'en', fallback chain is ['es', 'de']
        # en not available -> try es -> es not available -> try de -> de available!
        result = fetcher.select_best_language(
            available,
            preferred="en",
            fallback_languages=["es", "de"],
            fallback_to_english=False  # Don't auto-fallback to English
        )

        assert result is not None, "Should find de in fallback chain"
        assert result.code == "de", f"Expected 'de' but got '{result.code}'"
        assert result.is_auto_generated is False, "Should be manual captions"

    @pytest.mark.fast
    def test_unicode_handling_german_umlauts(self):
        """Test Unicode handling: German umlauts (ä, ö, ü, ß) in caption text."""
        # German text with umlauts
        german_text = "Größe und Bäume: Süßigkeiten für Köln"

        segments = [
            CaptionSegment(
                index=0,
                start_time=0.0,
                end_time=5.0,
                text=german_text,
                source_file="test_vid"
            )
        ]

        result = CaptionResult(
            video_id="test_vid",
            segments=segments,
            language="de",
            is_auto_generated=False,
            format_source="vtt"
        )

        # Verify umlauts preserved correctly
        assert "Größe" in result.text, "German 'Größe' should be preserved"
        assert "Bäume" in result.text, "German 'Bäume' should be preserved"
        assert "Süßigkeiten" in result.text, "German 'Süßigkeiten' should be preserved"
        assert "ö" in result.text and "ä" in result.text and "ü" in result.text
        assert "ß" in result.text, "German sharp s (ß) should be preserved"
        assert result.language == "de"

    @pytest.mark.fast
    def test_unicode_handling_chinese_characters(self):
        """Test Unicode handling: Chinese characters (simplified and traditional)."""
        # Mix of simplified and traditional Chinese
        chinese_text = "这是简体中文。這是繁體中文。你好世界！"

        segments = [
            CaptionSegment(
                index=0,
                start_time=0.0,
                end_time=5.0,
                text=chinese_text,
                source_file="test_vid"
            )
        ]

        result = CaptionResult(
            video_id="test_vid",
            segments=segments,
            language="zh",
            is_auto_generated=False,
            format_source="vtt"
        )

        # Verify Chinese characters preserved
        assert "简体中文" in result.text, "Simplified Chinese should be preserved"
        assert "繁體中文" in result.text, "Traditional Chinese should be preserved"
        assert "你好世界" in result.text, "Common Chinese characters preserved"
        assert result.language == "zh"
        assert len(result.segments) == 1

    @pytest.mark.fast
    def test_unicode_handling_arabic_rtl_text(self):
        """Test Unicode handling: Arabic right-to-left (RTL) text."""
        # Arabic text (reads right-to-left)
        arabic_text = "مرحبا بالعالم! هذا نص عربي للاختبار."

        segments = [
            CaptionSegment(
                index=0,
                start_time=0.0,
                end_time=5.0,
                text=arabic_text,
                source_file="test_vid"
            )
        ]

        result = CaptionResult(
            video_id="test_vid",
            segments=segments,
            language="ar",
            is_auto_generated=False,
            format_source="vtt"
        )

        # Verify Arabic text preserved (RTL handling is display-level, not storage)
        assert "مرحبا" in result.text, "Arabic greeting should be preserved"
        assert "العالم" in result.text, "Arabic word for 'world' should be preserved"
        assert result.language == "ar"
        # Arabic has specific character range U+0600-U+06FF
        arabic_chars = [c for c in result.text if '\u0600' <= c <= '\u06FF']
        assert len(arabic_chars) > 10, f"Should have Arabic chars, found {len(arabic_chars)}"

    @pytest.mark.fast
    def test_mixed_language_video_preferred_en_returned(self):
        """Test mixed-language video: captions in en + es available, preferred en returned."""
        fetcher = CaptionFetcher()

        # Video has both English and Spanish captions available
        available = [
            AvailableLanguage("en", "English", False),
            AvailableLanguage("es", "Spanish", False),
            AvailableLanguage("en", "English (auto)", True),  # Also auto-generated English
        ]

        # Preferred is English
        result = fetcher.select_best_language(
            available,
            preferred="en",
            fallback_languages=["es", "de"],
            prefer_manual=True
        )

        assert result is not None, "Should find English"
        assert result.code == "en", f"Preferred 'en' should be selected, got '{result.code}'"
        assert result.is_auto_generated is False, "Should prefer manual over auto"

    @pytest.mark.fast
    def test_language_code_normalization_uppercase_treated_same_as_lowercase(self):
        """Test language code normalization: 'EN' treated same as 'en'."""
        fetcher = CaptionFetcher()

        # Available languages with uppercase codes (unusual but possible)
        available = [
            AvailableLanguage("EN", "English", False),
            AvailableLanguage("ES", "Spanish", False),
        ]

        # Use lowercase 'en' in preferred
        result = fetcher.select_best_language(
            available,
            preferred="en",  # lowercase
            fallback_languages=["es", "de"]
        )

        assert result is not None, "Should match 'EN' from 'en'"
        # The code returned should be what's in available (uppercase)
        assert result.code.lower() == "en", "Should match English regardless of case"

        # Also test lowercase available with uppercase preferred
        available_lower = [
            AvailableLanguage("en", "English", False),
            AvailableLanguage("es", "Spanish", False),
        ]

        result2 = fetcher.select_best_language(
            available_lower,
            preferred="EN",  # uppercase
            fallback_languages=["ES", "DE"]
        )

        assert result2 is not None, "Should match 'en' from 'EN'"
        assert result2.code.lower() == "en", "Case-insensitive matching should work"

    @pytest.mark.fast
    def test_empty_fallback_languages_defaults_to_en_gracefully(self):
        """Test empty fallback_languages defaults to 'en' gracefully.

        When fallback_languages is [] (empty list), the system should
        still fall back to English via the fallback_to_english parameter.
        """
        fetcher = CaptionFetcher()

        # Spanish preferred, but only English available
        available = [
            AvailableLanguage("en", "English", False),
            AvailableLanguage("fr", "French", False),
        ]

        # Empty fallback_languages, but fallback_to_english is True (default)
        result = fetcher.select_best_language(
            available,
            preferred="es",
            fallback_languages=[],  # Empty!
            fallback_to_english=True  # This should kick in
        )

        assert result is not None, "Should fall back to English"
        assert result.code == "en", "With empty fallback_languages, should use English fallback"

        # Also verify that when fallback_languages is empty AND fallback_to_english=False,
        # it falls back to any available
        result2 = fetcher.select_best_language(
            available,
            preferred="es",
            fallback_languages=[],
            fallback_to_english=False
        )

        assert result2 is not None, "Should fall back to any available"
        # First in sorted list (en before fr alphabetically)
        assert result2.code in ["en", "fr"], "Should return some available language"


class TestMultilingualCaptionValidation:
    """Additional tests for language validation with multilingual content."""

    @pytest.mark.fast
    def test_is_valid_language_code_for_multilingual_set(self):
        """Test is_valid_language_code for a variety of language codes."""
        # Common languages used in multilingual projects
        valid_codes = ['en', 'es', 'de', 'fr', 'pt', 'zh', 'ja', 'ko', 'ar', 'hi', 'ru']

        for code in valid_codes:
            assert is_valid_language_code(code), f"'{code}' should be valid"
            assert is_valid_language_code(code.upper()), f"'{code.upper()}' should be valid (case insensitive)"

    @pytest.mark.fast
    def test_validate_language_config_multilingual_fallback_chain(self):
        """Test validate_language_config with a realistic multilingual fallback chain."""
        # Realistic config for Spanish content with Portuguese/English fallbacks
        issues = validate_language_config(
            preferred_language='es',
            fallback_languages=['pt', 'en', 'fr'],
            raise_on_error=False
        )

        assert len(issues) == 0, f"Valid config should have no issues: {issues}"

    @pytest.mark.fast
    def test_validate_language_config_warns_duplicate_preferred_in_fallback(self):
        """Test that having preferred language in fallback generates warning."""
        issues = validate_language_config(
            preferred_language='en',
            fallback_languages=['es', 'en', 'fr'],  # 'en' is redundant
            raise_on_error=False
        )

        assert len(issues) == 1, "Should have exactly one warning"
        assert 'redundant' in issues[0].lower() or 'warning' in issues[0].lower()


class TestMultilingualCaptionSegments:
    """Test caption segment handling with multilingual content."""

    @pytest.mark.fast
    def test_mixed_unicode_segments(self):
        """Test CaptionResult with segments containing different Unicode scripts."""
        segments = [
            CaptionSegment(0, 0.0, 3.0, "Hello World", "vid1"),  # English
            CaptionSegment(1, 3.0, 6.0, "Hallo Welt, größte Freude!", "vid1"),  # German
            CaptionSegment(2, 6.0, 9.0, "你好世界", "vid1"),  # Chinese
            CaptionSegment(3, 9.0, 12.0, "مرحبا بالعالم", "vid1"),  # Arabic
        ]

        result = CaptionResult(
            video_id="vid1",
            segments=segments,
            language="mul",  # 'mul' is ISO 639-2 for multiple languages
            is_auto_generated=False,
            format_source="vtt"
        )

        # All text should be accessible
        full_text = result.text
        assert "Hello World" in full_text
        assert "größte" in full_text
        assert "你好世界" in full_text
        assert "مرحبا" in full_text

        # Duration should still be calculated correctly
        assert result.duration == 12.0

    @pytest.mark.fast
    def test_segment_serialization_preserves_unicode(self):
        """Test that to_dict() preserves Unicode characters correctly."""
        segment = CaptionSegment(
            index=0,
            start_time=0.0,
            end_time=5.0,
            text="Größe 中文 مرحبا",
            source_file="test_vid"
        )

        data = segment.to_dict()

        assert data['text'] == "Größe 中文 مرحبا"
        # Each script should be preserved
        assert "Größe" in data['text']  # German
        assert "中文" in data['text']    # Chinese
        assert "مرحبا" in data['text']   # Arabic
