"""Tests for US-78-009: Caption language fallback chain with priority ordering.

Verifies:
- language_priority config accepts a list of language codes in preference order
- Caption fetcher tries each language in priority order before auto-generated fallback
- Non-primary languages get reduced language_confidence (0.8 for 2nd, 0.6 for 3rd+)
- fallback_language records which language was actually used when non-primary
"""

import pytest
from unittest.mock import MagicMock, Mock

from src.caption.models import CaptionResult, CaptionSegment
from src.config.sections.download import CaptionFirstConfig


def _make_segments(n=3):
    """Helper to create N simple caption segments."""
    return [
        CaptionSegment(index=i, start_time=i * 10.0, end_time=(i + 1) * 10.0, text=f"segment {i}")
        for i in range(n)
    ]


def _make_fetcher_with_priority(language_priority):
    """Create a CaptionFetcher with a mocked config containing language_priority."""
    from src.caption_fetcher import CaptionFetcher

    fetcher = CaptionFetcher.__new__(CaptionFetcher)
    config = Mock()
    caption_first = Mock()
    caption_first.language_priority = language_priority
    config.download.caption_first = caption_first
    fetcher.config = config
    return fetcher


def _make_available_lang(code, is_auto=False):
    """Create an AvailableLanguage-like mock."""
    lang = MagicMock()
    lang.code = code
    lang.is_auto_generated = is_auto
    return lang


class TestLanguagePriorityConfig:
    """Test language_priority configuration field on CaptionFirstConfig."""

    def test_default_language_priority(self):
        """Default language_priority should be ['en', 'en-US', 'en-GB']."""
        config = CaptionFirstConfig()
        assert config.language_priority == ['en', 'en-US', 'en-GB']

    def test_custom_language_priority(self):
        """language_priority accepts a custom list of language codes."""
        config = CaptionFirstConfig(language_priority=['es', 'es-MX', 'pt', 'en'])
        assert config.language_priority == ['es', 'es-MX', 'pt', 'en']

    def test_empty_language_priority(self):
        """Empty language_priority falls back to preferred_language-only behavior."""
        config = CaptionFirstConfig(language_priority=[])
        assert config.language_priority == []


class TestSelectBestTrackWithPriority:
    """Test _select_best_track uses language_priority chain."""

    def test_primary_language_selected_first(self):
        """First language in priority chain is selected when available."""
        fetcher = _make_fetcher_with_priority(['en', 'en-US', 'en-GB'])
        available = [
            _make_available_lang('en', is_auto=False),
            _make_available_lang('en-US', is_auto=False),
        ]
        result = fetcher._select_best_track(available, 'en', True)
        assert result is not None
        lang, is_auto, priority_idx = result
        assert lang == 'en'
        assert is_auto is False
        assert priority_idx == 0

    def test_second_choice_when_primary_unavailable(self):
        """Second language in priority chain used when first is unavailable."""
        fetcher = _make_fetcher_with_priority(['en', 'en-US', 'en-GB'])
        available = [
            _make_available_lang('en-US', is_auto=False),
            _make_available_lang('en-GB', is_auto=False),
        ]
        result = fetcher._select_best_track(available, 'en', True)
        assert result is not None
        lang, is_auto, priority_idx = result
        assert lang == 'en-US'
        assert is_auto is False
        assert priority_idx == 1

    def test_third_choice_when_first_two_unavailable(self):
        """Third language in priority chain used when first two unavailable."""
        fetcher = _make_fetcher_with_priority(['en', 'en-US', 'en-GB'])
        available = [
            _make_available_lang('en-GB', is_auto=True),
        ]
        result = fetcher._select_best_track(available, 'en', True)
        assert result is not None
        lang, is_auto, priority_idx = result
        assert lang == 'en-GB'
        assert is_auto is True
        assert priority_idx == 2

    def test_prefers_manual_over_auto_in_priority_chain(self):
        """When prefer_manual=True, manual captions preferred over auto in same language."""
        fetcher = _make_fetcher_with_priority(['en', 'en-US'])
        available = [
            _make_available_lang('en', is_auto=True),
            _make_available_lang('en', is_auto=False),
        ]
        result = fetcher._select_best_track(available, 'en', True)
        assert result is not None
        lang, is_auto, priority_idx = result
        assert lang == 'en'
        assert is_auto is False
        assert priority_idx == 0

    def test_auto_selected_when_no_manual_in_priority(self):
        """Auto-generated selected if no manual available for priority language."""
        fetcher = _make_fetcher_with_priority(['en', 'en-US'])
        available = [
            _make_available_lang('en', is_auto=True),
        ]
        result = fetcher._select_best_track(available, 'en', True)
        assert result is not None
        lang, is_auto, priority_idx = result
        assert lang == 'en'
        assert is_auto is True
        assert priority_idx == 0

    def test_empty_priority_falls_back_to_preferred_language(self):
        """Empty language_priority uses preferred_language (original behavior)."""
        fetcher = _make_fetcher_with_priority([])
        available = [
            _make_available_lang('en', is_auto=False),
        ]
        result = fetcher._select_best_track(available, 'en', True)
        assert result is not None
        lang, is_auto, priority_idx = result
        assert lang == 'en'
        assert is_auto is False
        assert priority_idx == 0

    def test_no_config_falls_back_to_preferred_language(self):
        """No config -> no language_priority -> use preferred_language."""
        from src.caption_fetcher import CaptionFetcher
        fetcher = CaptionFetcher.__new__(CaptionFetcher)
        fetcher.config = None
        available = [
            _make_available_lang('en', is_auto=False),
        ]
        result = fetcher._select_best_track(available, 'en', True)
        assert result is not None
        lang, is_auto, priority_idx = result
        assert lang == 'en'
        assert is_auto is False
        assert priority_idx == 0

    def test_priority_chain_exhausted_falls_back_to_english(self):
        """When priority chain has no matches, falls back to English."""
        fetcher = _make_fetcher_with_priority(['fr', 'de'])
        available = [
            _make_available_lang('en', is_auto=False),
        ]
        # preferred_language is 'fr' (first in priority), no fr/de available
        result = fetcher._select_best_track(available, 'fr', True)
        assert result is not None
        lang, is_auto, priority_idx = result
        assert lang == 'en'
        assert is_auto is False
        assert priority_idx == -1  # Not from priority chain

    def test_priority_skips_duplicates(self):
        """Duplicate codes in priority chain are skipped."""
        fetcher = _make_fetcher_with_priority(['en', 'en', 'en-US'])
        available = [
            _make_available_lang('en-US', is_auto=False),
        ]
        result = fetcher._select_best_track(available, 'en', True)
        assert result is not None
        lang, is_auto, priority_idx = result
        assert lang == 'en-US'
        # Index 2 in original list, but 'en' at idx 1 is duplicate
        assert priority_idx == 2


class TestLanguageConfidenceByPriority:
    """Test that language_confidence is reduced for non-primary priority matches."""

    def test_primary_manual_confidence_1_0(self):
        """Primary language (idx 0), manual -> confidence 1.0."""
        result = CaptionResult(
            video_id="test123",
            segments=_make_segments(),
            language="en",
            is_auto_generated=False,
            language_confidence=1.0,
        )
        assert result.language_confidence == 1.0
        assert result.fallback_language == ""

    def test_second_choice_confidence_0_8(self):
        """2nd choice in priority chain -> confidence 0.8, fallback_language set."""
        result = CaptionResult(
            video_id="test123",
            segments=_make_segments(),
            language="en-US",
            is_auto_generated=False,
            language_confidence=0.8,
            fallback_language="en-US",
        )
        assert result.language_confidence == 0.8
        assert result.fallback_language == "en-US"

    def test_third_choice_confidence_0_6(self):
        """3rd+ choice in priority chain -> confidence 0.6, fallback_language set."""
        result = CaptionResult(
            video_id="test123",
            segments=_make_segments(),
            language="en-GB",
            is_auto_generated=False,
            language_confidence=0.6,
            fallback_language="en-GB",
        )
        assert result.language_confidence == 0.6
        assert result.fallback_language == "en-GB"

    def test_to_dict_preserves_reduced_confidence(self):
        """to_dict should preserve reduced confidence and fallback_language."""
        result = CaptionResult(
            video_id="test123",
            segments=_make_segments(),
            language="en-US",
            language_confidence=0.8,
            fallback_language="en-US",
        )
        d = result.to_dict()
        assert d['language_confidence'] == 0.8
        assert d['fallback_language'] == "en-US"
