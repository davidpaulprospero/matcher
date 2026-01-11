"""
Tests for SRT timestamp utility functions in src/utils.py

Tests timestamp parsing and formatting for SRT files:
- parse_srt_timestamp() - convert SRT timestamp to seconds
- format_srt_timestamp() - convert seconds to SRT format
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import parse_srt_timestamp, format_srt_timestamp


class TestParseSrtTimestamp:
    """Test parse_srt_timestamp() function"""

    def test_parse_simple_timestamp(self):
        """Test parsing simple timestamp"""
        timestamp = "00:00:05,500"
        result = parse_srt_timestamp(timestamp)

        assert result == 5.5

    def test_parse_timestamp_with_minutes(self):
        """Test parsing timestamp with minutes"""
        timestamp = "00:01:30,250"
        result = parse_srt_timestamp(timestamp)

        assert result == 90.25

    def test_parse_timestamp_with_hours(self):
        """Test parsing timestamp with hours"""
        timestamp = "01:00:00,000"
        result = parse_srt_timestamp(timestamp)

        assert result == 3600.0

    def test_parse_complex_timestamp(self):
        """Test parsing complex timestamp"""
        timestamp = "01:23:45,678"
        result = parse_srt_timestamp(timestamp)

        # 1 hour + 23 min + 45.678 sec = 3600 + 1380 + 45.678
        expected = 1 * 3600 + 23 * 60 + 45.678
        assert abs(result - expected) < 0.001

    def test_parse_timestamp_zero(self):
        """Test parsing zero timestamp"""
        timestamp = "00:00:00,000"
        result = parse_srt_timestamp(timestamp)

        assert result == 0.0

    def test_parse_timestamp_with_spaces(self):
        """Test parsing timestamp with extra spaces"""
        timestamp = "  00:01:30,500  "
        result = parse_srt_timestamp(timestamp)

        assert result == 90.5

    def test_parse_timestamp_decimal_notation(self):
        """Test parsing timestamp with period instead of comma"""
        # parse_srt_timestamp replaces comma with period
        timestamp = "00:00:05.500"
        result = parse_srt_timestamp(timestamp)

        assert result == 5.5

    def test_parse_large_timestamp(self):
        """Test parsing large timestamp (>10 hours)"""
        timestamp = "12:34:56,789"
        result = parse_srt_timestamp(timestamp)

        expected = 12 * 3600 + 34 * 60 + 56.789
        assert abs(result - expected) < 0.001

    def test_parse_millisecond_precision(self):
        """Test parsing with millisecond precision"""
        timestamp = "00:00:00,001"
        result = parse_srt_timestamp(timestamp)

        assert result == 0.001

    def test_parse_fractional_seconds(self):
        """Test parsing fractional seconds"""
        timestamp = "00:00:12,345"
        result = parse_srt_timestamp(timestamp)

        assert abs(result - 12.345) < 0.0001


class TestFormatSrtTimestamp:
    """Test format_srt_timestamp() function"""

    def test_format_simple_seconds(self):
        """Test formatting simple seconds"""
        seconds = 5.5
        result = format_srt_timestamp(seconds)

        assert result == "00:00:05,500"

    def test_format_zero_seconds(self):
        """Test formatting zero seconds"""
        seconds = 0.0
        result = format_srt_timestamp(seconds)

        assert result == "00:00:00,000"

    def test_format_minutes_and_seconds(self):
        """Test formatting minutes and seconds"""
        seconds = 90.25
        result = format_srt_timestamp(seconds)

        assert result == "00:01:30,250"

    def test_format_hours_minutes_seconds(self):
        """Test formatting hours, minutes, and seconds"""
        seconds = 3661.5  # 1 hour, 1 min, 1.5 sec
        result = format_srt_timestamp(seconds)

        assert result == "01:01:01,500"

    def test_format_millisecond_precision(self):
        """Test formatting with millisecond precision"""
        seconds = 12.345
        result = format_srt_timestamp(seconds)

        assert result == "00:00:12,345"

    def test_format_uses_comma_not_period(self):
        """Test that format uses comma, not period"""
        seconds = 5.5
        result = format_srt_timestamp(seconds)

        assert ',' in result
        # After the last colon, before the digits
        assert result.count(',') == 1

    def test_format_complex_timestamp(self):
        """Test formatting complex timestamp"""
        seconds = 1 * 3600 + 23 * 60 + 45.678
        result = format_srt_timestamp(seconds)

        assert result == "01:23:45,678"

    def test_format_large_hours(self):
        """Test formatting large hours value"""
        seconds = 12 * 3600 + 34 * 60 + 56.789
        result = format_srt_timestamp(seconds)

        assert result.startswith("12:34:56")

    def test_format_fractional_milliseconds(self):
        """Test formatting with fractional milliseconds"""
        seconds = 0.001
        result = format_srt_timestamp(seconds)

        assert result == "00:00:00,001"

    def test_format_no_milliseconds(self):
        """Test formatting whole seconds"""
        seconds = 60.0
        result = format_srt_timestamp(seconds)

        assert result == "00:01:00,000"

    def test_format_padding_hours(self):
        """Test zero-padding for hours"""
        seconds = 3600  # 1 hour
        result = format_srt_timestamp(seconds)

        assert result.startswith("01:")

    def test_format_padding_minutes(self):
        """Test zero-padding for minutes"""
        seconds = 5 * 60  # 5 minutes
        result = format_srt_timestamp(seconds)

        assert ":05:" in result

    def test_format_padding_seconds(self):
        """Test zero-padding for seconds"""
        seconds = 5.5
        result = format_srt_timestamp(seconds)

        assert result.startswith("00:00:05")


class TestTimestampRoundtrip:
    """Test roundtrip conversion (parse -> format -> parse)"""

    def test_roundtrip_simple(self):
        """Test roundtrip with simple timestamp"""
        original = "00:00:05,500"

        seconds = parse_srt_timestamp(original)
        formatted = format_srt_timestamp(seconds)
        reparsed = parse_srt_timestamp(formatted)

        assert abs(seconds - reparsed) < 0.0001

    def test_roundtrip_complex(self):
        """Test roundtrip with complex timestamp"""
        original = "01:23:45,678"

        seconds = parse_srt_timestamp(original)
        formatted = format_srt_timestamp(seconds)

        # Should match original format
        assert formatted == original

    def test_roundtrip_zero(self):
        """Test roundtrip with zero"""
        original = "00:00:00,000"

        seconds = parse_srt_timestamp(original)
        formatted = format_srt_timestamp(seconds)

        assert formatted == original

    def test_roundtrip_milliseconds(self):
        """Test roundtrip preserves milliseconds"""
        original = "00:00:12,345"

        seconds = parse_srt_timestamp(original)
        formatted = format_srt_timestamp(seconds)

        assert formatted == original

    def test_roundtrip_hours(self):
        """Test roundtrip with hours"""
        original = "12:00:00,000"

        seconds = parse_srt_timestamp(original)
        formatted = format_srt_timestamp(seconds)

        assert formatted == original

    def test_roundtrip_precision(self):
        """Test that roundtrip maintains precision"""
        test_cases = [
            "00:00:01,001",
            "00:01:00,100",
            "01:00:00,999",
            "01:23:45,678",
            "00:00:00,123",
        ]

        for timestamp in test_cases:
            seconds = parse_srt_timestamp(timestamp)
            formatted = format_srt_timestamp(seconds)
            assert formatted == timestamp, f"Failed for {timestamp}"


class TestTimestampEdgeCases:
    """Test edge cases for timestamp functions"""

    def test_parse_very_large_hours(self):
        """Test parsing very large hour values"""
        timestamp = "99:59:59,999"
        result = parse_srt_timestamp(timestamp)

        expected = 99 * 3600 + 59 * 60 + 59.999
        assert abs(result - expected) < 0.001

    def test_format_very_large_seconds(self):
        """Test formatting very large seconds value"""
        seconds = 100 * 3600  # 100 hours
        result = format_srt_timestamp(seconds)

        assert result.startswith("100:")

    def test_parse_minimal_milliseconds(self):
        """Test parsing minimal milliseconds"""
        timestamp = "00:00:00,001"
        result = parse_srt_timestamp(timestamp)

        assert result == 0.001

    def test_format_tiny_fraction(self):
        """Test formatting very small fraction"""
        seconds = 0.0001
        result = format_srt_timestamp(seconds)

        # Should round to nearest millisecond
        assert "00:00:00" in result

    def test_parse_with_leading_zeros(self):
        """Test parsing with extra leading zeros"""
        timestamp = "000:000:005,500"
        result = parse_srt_timestamp(timestamp)

        assert result == 5.5

    def test_format_fractional_hours(self):
        """Test formatting when conversion creates fractional values"""
        # 90 minutes = 1.5 hours, should format as 01:30:00
        seconds = 90 * 60
        result = format_srt_timestamp(seconds)

        assert result == "01:30:00,000"

    def test_parse_max_milliseconds(self):
        """Test parsing maximum milliseconds (999)"""
        timestamp = "00:00:00,999"
        result = parse_srt_timestamp(timestamp)

        assert result == 0.999

    def test_format_round_to_millisecond(self):
        """Test that formatting rounds to milliseconds"""
        # 0.0001 seconds should round to 0.000
        seconds = 0.0001
        result = format_srt_timestamp(seconds)

        # Check it rounds to 3 decimal places
        parts = result.split(',')
        assert len(parts[1]) == 3


class TestTimestampConsistency:
    """Test consistency between parse and format"""

    def test_format_parse_inverse(self):
        """Test that format and parse are inverse operations"""
        test_seconds = [0.0, 1.0, 60.0, 3600.0, 3661.5, 12345.678]

        for seconds in test_seconds:
            formatted = format_srt_timestamp(seconds)
            parsed = parse_srt_timestamp(formatted)

            # Should be very close (within millisecond precision)
            assert abs(seconds - parsed) < 0.001

    def test_multiple_roundtrips(self):
        """Test that multiple roundtrips don't accumulate error"""
        seconds = 123.456

        for _ in range(5):
            formatted = format_srt_timestamp(seconds)
            seconds = parse_srt_timestamp(formatted)

        # After 5 roundtrips, should still match original
        assert abs(seconds - 123.456) < 0.001

    def test_parse_different_notations(self):
        """Test parsing different timestamp notations"""
        # With comma
        result1 = parse_srt_timestamp("00:00:05,500")
        # With period (gets converted internally)
        result2 = parse_srt_timestamp("00:00:05.500")

        assert result1 == result2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
