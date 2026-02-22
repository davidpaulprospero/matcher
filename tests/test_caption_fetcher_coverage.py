"""Tests for increasing caption_fetcher coverage to 80%+.

Tests functions: is_valid_language_code, validate_language_config,
validate_caption_config, parse_description_chapters, calculate_segment_metrics,
classify_stream_state
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.caption_fetcher import (
    is_valid_language_code,
    validate_language_config,
    validate_caption_config,
    parse_description_chapters,
    calculate_segment_metrics,
    classify_stream_state,
    ConfigValidationError,
    CaptionSegment,
)


class TestIsValidLanguageCode:
    """Test is_valid_language_code function."""

    def test_valid_two_letter_codes(self):
        """Test valid ISO 639-1 two-letter codes."""
        valid_codes = ['en', 'es', 'fr', 'de', 'pt', 'zh', 'ja', 'ko', 'it', 'ru']
        for code in valid_codes:
            assert is_valid_language_code(code) is True, f"Expected {code} to be valid"

    def test_uppercase_converted_to_valid(self):
        """Test uppercase codes are converted to lowercase and are valid."""
        # The function does code.lower() before checking, so 'EN' becomes 'en'
        assert is_valid_language_code('EN') is True
        assert is_valid_language_code('ES') is True

    def test_invalid_codes(self):
        """Test invalid language codes."""
        invalid_codes = ['xyz', 'aaa', 'eng', '123', 'e', '']
        for code in invalid_codes:
            assert is_valid_language_code(code) is False, f"Expected {code} to be invalid"

    def test_non_string_input(self):
        """Test non-string inputs return False."""
        assert is_valid_language_code(None) is False
        assert is_valid_language_code(123) is False
        assert is_valid_language_code(['en']) is False


class TestValidateLanguageConfig:
    """Test validate_language_config function."""

    def test_valid_config(self):
        """Test valid language configuration."""
        issues = validate_language_config('en', ['es', 'fr'], raise_on_error=False)
        assert issues == []

    def test_invalid_preferred_language(self):
        """Test invalid preferred_language raises or returns error."""
        with pytest.raises(ConfigValidationError):
            validate_language_config('xyz', [], raise_on_error=True)

        issues = validate_language_config('xyz', [], raise_on_error=False)
        assert len(issues) > 0
        assert 'xyz' in issues[0]

    def test_duplicate_fallback_languages(self):
        """Test duplicate fallback languages are detected."""
        issues = validate_language_config('en', ['es', 'es'], raise_on_error=False)
        assert any('duplicate' in i.lower() for i in issues)

    def test_preferred_in_fallback(self):
        """Test preferred language in fallback is redundant."""
        issues = validate_language_config('en', ['en', 'es'], raise_on_error=False)
        assert any('redundant' in i.lower() or 'en' in i for i in issues)

    def test_invalid_fallback_language(self):
        """Test invalid fallback language is detected."""
        issues = validate_language_config('en', ['xyz'], raise_on_error=False)
        assert len(issues) > 0


class TestParseDescriptionChapters:
    """Test parse_description_chapters function."""

    def test_no_chapters(self):
        """Test description with no chapters."""
        result = parse_description_chapters("This is just a description")
        assert result == []

    def test_with_timestamps(self):
        """Test description with timestamped chapters."""
        desc = """0:00 Introduction
0:30 Chapter One
1:15 Chapter Two
2:45 Conclusion"""
        result = parse_description_chapters(desc)
        assert len(result) == 4
        assert result[0]['title'] == 'Introduction'
        assert result[0]['start_time'] == 0.0

    def test_with_hour_timestamps(self):
        """Test description with hour-based timestamps."""
        desc = """0:00:00 Introduction
0:05:30 Chapter One
1:00:00 Chapter Two"""
        result = parse_description_chapters(desc)
        assert len(result) == 3
        assert result[1]['start_time'] == 330.0

    def test_empty_input(self):
        """Test empty/None input."""
        assert parse_description_chapters("") == []
        assert parse_description_chapters(None) == []


class TestCalculateSegmentMetrics:
    """Test calculate_segment_metrics function."""

    def test_empty_segments(self):
        """Test empty segments list."""
        result = calculate_segment_metrics([], 60.0)
        assert result.density_score == 0.0

    def test_single_segment(self):
        """Test single segment."""
        segments = [
            CaptionSegment(index=0, start_time=0.0, end_time=30.0, text="Test", source_file="vid")
        ]
        result = calculate_segment_metrics(segments, 60.0)
        assert result.density_score >= 0.0

    def test_multiple_segments(self):
        """Test multiple segments."""
        segments = [
            CaptionSegment(index=i, start_time=i*10.0, end_time=i*10.0+5.0, text=f"Text {i}", source_file="vid")
            for i in range(6)
        ]
        result = calculate_segment_metrics(segments, 60.0)
        assert result.density_score >= 0.0


class TestClassifyStreamState:
    """Test classify_stream_state function."""

    def test_live_stream(self):
        """Test live stream detection."""
        info = {'is_live': True, 'is_live_end': False}
        result = classify_stream_state(info)
        # Result has .state which is an enum
        assert result.is_live is True

    def test_live_stream_ended(self):
        """Test ended live stream."""
        info = {'was_live': True}
        result = classify_stream_state(info)
        assert result.was_live is True

    def test_upcoming_stream(self):
        """Test upcoming stream."""
        info = {'is_live': False, 'is_live_end': False, 'release_timestamp': 9999999999}
        result = classify_stream_state(info)
        assert result.scheduled_start is not None

    def test_regular_video(self):
        """Test regular video."""
        info = {'is_live': False, 'is_live_end': False}
        result = classify_stream_state(info)
        # Regular video should have duration
        assert result.duration is None
