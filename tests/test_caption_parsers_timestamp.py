"""
Tests for the canonical parse_timestamp function in src.caption.parsers.

Verifies the consolidated timestamp parsing handles all supported formats
and edge cases (US-66-002).
"""

import pytest
from src.caption.parsers import parse_timestamp


@pytest.mark.fast
class TestParseTimestampCanonical:
    """Test the canonical parse_timestamp function covers all formats."""

    # --- Standard formats ---

    def test_vtt_format(self):
        """HH:MM:SS.mmm (VTT)"""
        assert parse_timestamp("01:30:45.123") == 5445.123

    def test_srt_format(self):
        """HH:MM:SS,mmm (SRT) - comma as decimal separator"""
        assert parse_timestamp("01:30:45,123") == 5445.123

    def test_short_vtt_format(self):
        """MM:SS.mmm (VTT short)"""
        assert parse_timestamp("05:30.500") == 330.5

    def test_no_milliseconds(self):
        """HH:MM:SS without milliseconds"""
        assert parse_timestamp("01:30:45") == 5445.0

    def test_short_no_milliseconds(self):
        """MM:SS without milliseconds"""
        assert parse_timestamp("01:30") == 90.0

    def test_bare_float(self):
        """Bare float string (e.g. '123.456')"""
        assert parse_timestamp("123.456") == 123.456

    def test_bare_integer_string(self):
        """Bare integer string (e.g. '90')"""
        assert parse_timestamp("90") == 90.0

    # --- Edge cases required by acceptance criteria ---

    def test_none_input(self):
        """None input returns None"""
        assert parse_timestamp(None) is None

    def test_empty_string(self):
        """Empty string returns None"""
        assert parse_timestamp("") is None

    def test_whitespace_only(self):
        """Whitespace-only string returns None"""
        assert parse_timestamp("   ") is None

    def test_missing_hours(self):
        """MM:SS.mmm format (hours omitted) parses correctly"""
        assert parse_timestamp("00:01.000") == 1.0
        assert parse_timestamp("59:59.999") == 3599.999

    def test_overflow_milliseconds(self):
        """Milliseconds with more than 3 digits are truncated to 3"""
        # "12345" -> ljust(3,'0')[:3] -> "123" -> 123ms
        assert parse_timestamp("00:00:01.12345") == 1.123

    def test_single_digit_milliseconds(self):
        """Single-digit milliseconds are padded to 3 digits"""
        # "1" -> ljust(3,'0') -> "100" -> 100ms
        assert parse_timestamp("00:00:01.1") == 1.1

    def test_two_digit_milliseconds(self):
        """Two-digit milliseconds are padded to 3 digits"""
        # "12" -> ljust(3,'0') -> "120" -> 120ms
        assert parse_timestamp("00:00:01.12") == 1.12

    def test_zero_timestamp(self):
        """Zero timestamp parses to 0.0"""
        assert parse_timestamp("00:00:00.000") == 0.0

    def test_invalid_string(self):
        """Non-timestamp strings return None"""
        assert parse_timestamp("not a timestamp") is None

    def test_non_numeric_colons(self):
        """Non-numeric colon-separated values return None"""
        assert parse_timestamp("abc:def:ghi") is None

    def test_whitespace_around_timestamp(self):
        """Leading/trailing whitespace is stripped"""
        assert parse_timestamp("  00:00:01.000  ") == 1.0

    def test_large_hours(self):
        """Large hour values parse correctly"""
        assert parse_timestamp("100:00:00.000") == 360000.0

    def test_srt_comma_short_format(self):
        """MM:SS,mmm with comma separator"""
        assert parse_timestamp("05:30,500") == 330.5

    def test_negative_bare_float(self):
        """Negative bare float returns None (timestamps can't be negative)"""
        assert parse_timestamp("-1.5") is None

    def test_negative_integer(self):
        """Negative integer string returns None"""
        assert parse_timestamp("-60") is None

    def test_negative_timestamp_format(self):
        """Negative timestamp format returns None"""
        assert parse_timestamp("-00:01:00.000") is None


@pytest.mark.fast
class TestParseTimestampDelegation:
    """Verify that normalizer and fetcher delegate to canonical parse_timestamp."""

    def test_normalizer_delegates(self):
        """CaptionNormalizer.convert_timestamp_to_seconds uses canonical function"""
        from src.caption.normalizer import CaptionNormalizer
        # These should produce identical results
        assert CaptionNormalizer.convert_timestamp_to_seconds("01:30:45.123") == parse_timestamp("01:30:45.123")
        assert CaptionNormalizer.convert_timestamp_to_seconds("01:30:45,123") == parse_timestamp("01:30:45,123")
        assert CaptionNormalizer.convert_timestamp_to_seconds("05:30.500") == parse_timestamp("05:30.500")
        assert CaptionNormalizer.convert_timestamp_to_seconds("") == parse_timestamp("")
        assert CaptionNormalizer.convert_timestamp_to_seconds(None) == parse_timestamp(None)

    def test_fetcher_instance_delegates(self):
        """CaptionFetcher._parse_timestamp uses canonical function"""
        from src.caption_fetcher import CaptionFetcher
        fetcher = CaptionFetcher()
        assert fetcher._parse_timestamp("01:30:45.123") == parse_timestamp("01:30:45.123")
        assert fetcher._parse_timestamp("01:30:45,123") == parse_timestamp("01:30:45,123")
        assert fetcher._parse_timestamp("05:30.500") == parse_timestamp("05:30.500")
