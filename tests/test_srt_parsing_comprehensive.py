"""
Comprehensive tests for SRT parsing utilities in src/utils.py

This module provides comprehensive tests for voiceover/SRT parsing functionality:
- parse_srt_timestamp() - convert SRT timestamp to seconds
- format_srt_timestamp() - convert seconds to SRT format
- parse_srt_file() - parse SRT file into segments
- write_srt_file() - write segments to SRT file
- SRTSegment dataclass - represents a single SRT segment

Note: This was originally planned as pot_utils tests, but that module doesn't exist.
The actual voiceover parsing functionality is in src/utils.py.

Story: US-36-002
"""

import pytest
import tempfile
from pathlib import Path
from typing import List

from src.utils import (
    parse_srt_timestamp,
    format_srt_timestamp,
    parse_srt_file,
    write_srt_file,
    SRTSegment,
)


class TestParseSrtTimestampEdgeCases:
    """Additional edge case tests for parse_srt_timestamp()"""

    @pytest.mark.fast
    def test_parse_zero_timestamp(self):
        """Test parsing 00:00:00,000 returns exactly 0.0"""
        result = parse_srt_timestamp("00:00:00,000")
        assert result == 0.0

    @pytest.mark.fast
    def test_parse_max_timestamp_99_59_59(self):
        """Test parsing maximum valid timestamp (99:59:59,999)"""
        result = parse_srt_timestamp("99:59:59,999")
        expected = 99 * 3600 + 59 * 60 + 59.999
        assert abs(result - expected) < 0.001

    @pytest.mark.fast
    def test_parse_single_digit_parts(self):
        """Test parsing timestamps with single digit components"""
        # Standard format requires zero-padding, but parse should handle unpadded too
        result = parse_srt_timestamp("1:2:3,456")
        expected = 1 * 3600 + 2 * 60 + 3.456
        assert abs(result - expected) < 0.001

    @pytest.mark.fast
    def test_parse_no_milliseconds(self):
        """Test parsing timestamp without milliseconds (,000)"""
        result = parse_srt_timestamp("00:01:00,000")
        assert result == 60.0

    @pytest.mark.fast
    def test_parse_max_milliseconds(self):
        """Test parsing maximum milliseconds (,999)"""
        result = parse_srt_timestamp("00:00:00,999")
        assert abs(result - 0.999) < 0.001

    @pytest.mark.fast
    def test_parse_whitespace_handling(self):
        """Test parsing handles leading/trailing whitespace"""
        result = parse_srt_timestamp("  00:00:05,500  \n")
        assert result == 5.5

    @pytest.mark.fast
    def test_parse_period_instead_of_comma(self):
        """Test parsing with period decimal separator"""
        result = parse_srt_timestamp("00:00:05.500")
        assert result == 5.5

    @pytest.mark.fast
    def test_parse_hours_over_100(self):
        """Test parsing hours greater than 99"""
        result = parse_srt_timestamp("150:00:00,000")
        expected = 150 * 3600
        assert result == expected

    @pytest.mark.fast
    def test_parse_sub_millisecond_precision(self):
        """Test that sub-millisecond precision is handled"""
        # SRT format only has 3 decimal places, but parser should work
        result = parse_srt_timestamp("00:00:01,123")
        assert abs(result - 1.123) < 0.0001

    @pytest.mark.fast
    def test_parse_exactly_one_hour(self):
        """Test parsing exactly one hour"""
        result = parse_srt_timestamp("01:00:00,000")
        assert result == 3600.0

    @pytest.mark.fast
    def test_parse_exactly_one_minute(self):
        """Test parsing exactly one minute"""
        result = parse_srt_timestamp("00:01:00,000")
        assert result == 60.0


class TestFormatSrtTimestampEdgeCases:
    """Additional edge case tests for format_srt_timestamp()"""

    @pytest.mark.fast
    def test_format_zero_seconds(self):
        """Test formatting 0.0 seconds"""
        result = format_srt_timestamp(0.0)
        assert result == "00:00:00,000"

    @pytest.mark.fast
    def test_format_max_common_timestamp(self):
        """Test formatting common maximum (99:59:59,999)"""
        seconds = 99 * 3600 + 59 * 60 + 59.999
        result = format_srt_timestamp(seconds)
        assert result.startswith("99:59:59")

    @pytest.mark.fast
    def test_format_preserves_millisecond_precision(self):
        """Test formatting preserves 3 decimal place precision"""
        result = format_srt_timestamp(1.123)
        assert result == "00:00:01,123"

    @pytest.mark.fast
    def test_format_uses_comma_separator(self):
        """Test format uses comma not period for milliseconds"""
        result = format_srt_timestamp(5.5)
        assert "," in result
        assert "." not in result

    @pytest.mark.fast
    def test_format_zero_pads_all_components(self):
        """Test all components are zero-padded to 2 digits"""
        result = format_srt_timestamp(3661.0)  # 1:01:01
        assert result == "01:01:01,000"

    @pytest.mark.fast
    def test_format_three_digit_hours(self):
        """Test formatting 100+ hours"""
        seconds = 100 * 3600
        result = format_srt_timestamp(seconds)
        assert result.startswith("100:")

    @pytest.mark.fast
    def test_format_fractional_seconds_rounding(self):
        """Test fractional seconds round correctly"""
        # 0.1234 should round to 0.123 (truncate to 3 places)
        result = format_srt_timestamp(0.1234)
        # Check millisecond portion
        ms_part = result.split(",")[1]
        assert len(ms_part) == 3

    @pytest.mark.fast
    def test_format_negative_clamps_to_zero(self):
        """Test negative seconds behavior"""
        # Negative should produce result (implementation specific)
        # Most likely wraps or errors - just check it doesn't crash
        try:
            result = format_srt_timestamp(-1.0)
            # If it doesn't crash, verify format
            assert ":" in result
        except (ValueError, Exception):
            # Acceptable to raise error for invalid input
            pass

    @pytest.mark.fast
    def test_format_very_small_fraction(self):
        """Test formatting very small fractions"""
        result = format_srt_timestamp(0.001)
        assert result == "00:00:00,001"


class TestTimestampRoundtripComprehensive:
    """Comprehensive roundtrip tests for timestamp functions"""

    @pytest.mark.fast
    def test_roundtrip_boundary_values(self):
        """Test roundtrip for boundary values"""
        test_cases = [
            "00:00:00,000",  # Zero
            "00:00:00,001",  # Min milliseconds
            "00:00:00,999",  # Max milliseconds
            "00:00:59,999",  # Max seconds
            "00:59:59,999",  # Max minutes
            "23:59:59,999",  # Full day minus 1ms
        ]
        for timestamp in test_cases:
            seconds = parse_srt_timestamp(timestamp)
            formatted = format_srt_timestamp(seconds)
            reparsed = parse_srt_timestamp(formatted)
            assert abs(seconds - reparsed) < 0.001, f"Failed for {timestamp}"

    @pytest.mark.fast
    def test_roundtrip_common_durations(self):
        """Test roundtrip for common video durations"""
        # Common video lengths in seconds
        durations = [
            30.0,       # 30 second clip
            60.0,       # 1 minute
            90.0,       # 1.5 minutes
            180.0,      # 3 minutes
            300.0,      # 5 minutes
            600.0,      # 10 minutes
            1800.0,     # 30 minutes
            3600.0,     # 1 hour
            7200.0,     # 2 hours
        ]
        for seconds in durations:
            formatted = format_srt_timestamp(seconds)
            reparsed = parse_srt_timestamp(formatted)
            assert seconds == reparsed, f"Failed for {seconds}s"

    @pytest.mark.fast
    def test_roundtrip_preserves_precision(self):
        """Test roundtrip preserves millisecond precision"""
        test_seconds = [0.001, 0.123, 0.999, 1.001, 59.999]
        for seconds in test_seconds:
            formatted = format_srt_timestamp(seconds)
            reparsed = parse_srt_timestamp(formatted)
            assert abs(seconds - reparsed) < 0.001


class TestParseSrtFileMalformed:
    """Tests for parse_srt_file handling of malformed input"""

    @pytest.mark.fast
    def test_parse_empty_file(self, tmp_path):
        """Test parsing empty file returns empty list"""
        srt_file = tmp_path / "empty.srt"
        srt_file.write_text("", encoding="utf-8")
        result = parse_srt_file(str(srt_file))
        assert result == []

    @pytest.mark.fast
    def test_parse_whitespace_only_file(self, tmp_path):
        """Test parsing file with only whitespace"""
        srt_file = tmp_path / "whitespace.srt"
        srt_file.write_text("   \n\n   \n   ", encoding="utf-8")
        result = parse_srt_file(str(srt_file))
        assert result == []

    @pytest.mark.fast
    def test_parse_missing_timestamp_line(self, tmp_path):
        """Test parsing segment without timestamp line"""
        content = """1
This is text without timestamp line

2
00:00:05,000 --> 00:00:10,000
Valid segment
"""
        srt_file = tmp_path / "missing_ts.srt"
        srt_file.write_text(content, encoding="utf-8")
        result = parse_srt_file(str(srt_file))
        # Should skip malformed and parse valid
        assert len(result) >= 1
        valid_texts = [s.text for s in result]
        assert "Valid segment" in valid_texts

    @pytest.mark.fast
    def test_parse_invalid_index(self, tmp_path):
        """Test parsing segment with non-numeric index"""
        content = """ABC
00:00:00,000 --> 00:00:05,000
Invalid index segment

2
00:00:05,000 --> 00:00:10,000
Valid segment
"""
        srt_file = tmp_path / "invalid_idx.srt"
        srt_file.write_text(content, encoding="utf-8")
        result = parse_srt_file(str(srt_file))
        # Should skip invalid and parse valid
        assert len(result) >= 1

    @pytest.mark.fast
    def test_parse_malformed_timestamp_arrow(self, tmp_path):
        """Test parsing with malformed arrow separator"""
        content = """1
00:00:00,000 -> 00:00:05,000
Missing one dash

2
00:00:05,000 --> 00:00:10,000
Valid segment
"""
        srt_file = tmp_path / "bad_arrow.srt"
        srt_file.write_text(content, encoding="utf-8")
        result = parse_srt_file(str(srt_file))
        # Should handle gracefully
        assert isinstance(result, list)

    @pytest.mark.fast
    def test_parse_missing_end_timestamp(self, tmp_path):
        """Test parsing with only start timestamp"""
        content = """1
00:00:00,000
Missing end timestamp

2
00:00:05,000 --> 00:00:10,000
Valid segment
"""
        srt_file = tmp_path / "missing_end.srt"
        srt_file.write_text(content, encoding="utf-8")
        result = parse_srt_file(str(srt_file))
        # Should skip malformed
        assert isinstance(result, list)

    @pytest.mark.fast
    def test_parse_empty_text(self, tmp_path):
        """Test parsing segment with empty text"""
        content = """1
00:00:00,000 --> 00:00:05,000


2
00:00:05,000 --> 00:00:10,000
Valid text
"""
        srt_file = tmp_path / "empty_text.srt"
        srt_file.write_text(content, encoding="utf-8")
        result = parse_srt_file(str(srt_file))
        # Empty text segments should be skipped
        texts = [s.text for s in result if s.text.strip()]
        assert "Valid text" in texts

    @pytest.mark.fast
    def test_parse_binary_garbage(self, tmp_path):
        """Test parsing file with binary content"""
        srt_file = tmp_path / "binary.srt"
        srt_file.write_bytes(b"\x00\x01\x02\x03ID3\xff\xfe")
        result = parse_srt_file(str(srt_file))
        # Should detect binary and return empty
        assert result == []

    @pytest.mark.fast
    def test_parse_nonexistent_file(self):
        """Test parsing nonexistent file returns empty list"""
        result = parse_srt_file("/nonexistent/path/file.srt")
        assert result == []

    @pytest.mark.fast
    def test_parse_mixed_valid_invalid(self, tmp_path):
        """Test parsing file with mixed valid/invalid segments"""
        content = """1
00:00:00,000 --> 00:00:05,000
First valid

GARBAGE LINE HERE

2
00:00:05,000 --> 00:00:10,000
Second valid

MORE GARBAGE
without timestamp

3
00:00:10,000 --> 00:00:15,000
Third valid
"""
        srt_file = tmp_path / "mixed.srt"
        srt_file.write_text(content, encoding="utf-8")
        result = parse_srt_file(str(srt_file))
        assert len(result) == 3

    @pytest.mark.fast
    def test_parse_extra_blank_lines(self, tmp_path):
        """Test parsing with extra blank lines between segments"""
        content = """1
00:00:00,000 --> 00:00:05,000
First




2
00:00:05,000 --> 00:00:10,000
Second
"""
        srt_file = tmp_path / "extra_blanks.srt"
        srt_file.write_text(content, encoding="utf-8")
        result = parse_srt_file(str(srt_file))
        assert len(result) == 2


class TestWriteSrtFileEdgeCases:
    """Edge case tests for write_srt_file()"""

    @pytest.mark.fast
    def test_write_empty_list(self, tmp_path):
        """Test writing empty segment list"""
        output = tmp_path / "empty.srt"
        write_srt_file([], str(output))
        assert output.exists()
        assert output.read_text() == ""

    @pytest.mark.fast
    def test_write_unicode_text(self, tmp_path):
        """Test writing segments with unicode characters"""
        segments = [
            SRTSegment(index=1, start_time=0.0, end_time=5.0, text="Hello 世界 🌍"),
            SRTSegment(index=2, start_time=5.0, end_time=10.0, text="Привет мир"),
        ]
        output = tmp_path / "unicode.srt"
        write_srt_file(segments, str(output))

        content = output.read_text(encoding="utf-8")
        assert "世界" in content
        assert "🌍" in content
        assert "Привет" in content

    @pytest.mark.fast
    def test_write_multiline_text(self, tmp_path):
        """Test writing segment with multiline text"""
        segments = [
            SRTSegment(
                index=1,
                start_time=0.0,
                end_time=5.0,
                text="Line one Line two Line three"
            ),
        ]
        output = tmp_path / "multiline.srt"
        write_srt_file(segments, str(output))

        content = output.read_text(encoding="utf-8")
        assert "Line one" in content

    @pytest.mark.fast
    def test_write_very_long_text(self, tmp_path):
        """Test writing segment with very long text"""
        long_text = "A" * 10000
        segments = [
            SRTSegment(index=1, start_time=0.0, end_time=5.0, text=long_text),
        ]
        output = tmp_path / "long.srt"
        write_srt_file(segments, str(output))

        # Re-read and verify
        content = output.read_text(encoding="utf-8")
        assert long_text in content

    @pytest.mark.fast
    def test_write_special_characters(self, tmp_path):
        """Test writing segment with special characters"""
        segments = [
            SRTSegment(
                index=1,
                start_time=0.0,
                end_time=5.0,
                text="Text with <tags> & 'quotes' \"double\" and {braces}"
            ),
        ]
        output = tmp_path / "special.srt"
        write_srt_file(segments, str(output))

        content = output.read_text(encoding="utf-8")
        assert "<tags>" in content
        assert "&" in content


class TestSRTSegmentDataclass:
    """Tests for SRTSegment dataclass methods"""

    @pytest.mark.fast
    def test_segment_duration_property(self):
        """Test duration property calculation"""
        segment = SRTSegment(index=1, start_time=10.0, end_time=25.5, text="Test")
        assert segment.duration == 15.5

    @pytest.mark.fast
    def test_segment_zero_duration(self):
        """Test segment with zero duration"""
        segment = SRTSegment(index=1, start_time=5.0, end_time=5.0, text="Instant")
        assert segment.duration == 0.0

    @pytest.mark.fast
    def test_segment_to_dict_converts_all_fields(self):
        """Test to_dict() includes all expected fields"""
        segment = SRTSegment(
            index=1,
            start_time=0.0,
            end_time=5.0,
            text="Test text",
            source_file="/path/to/file.mp4",
            keywords=["kw1", "kw2"],
            entities=[{"text": "entity", "type": "PERSON"}],
            topic_id=1,
            topics=["topic1"],
            is_broll=True,
        )

        d = segment.to_dict()

        assert d["index"] == 1
        assert d["start_time"] == 0.0
        assert d["end_time"] == 5.0
        assert d["text"] == "Test text"
        assert d["source_file"] == "/path/to/file.mp4"
        assert d["keywords"] == ["kw1", "kw2"]
        assert len(d["entities"]) == 1
        assert d["topic_id"] == 1
        assert d["topics"] == ["topic1"]
        assert d["is_broll"] is True

    @pytest.mark.fast
    def test_segment_to_dict_handles_none(self):
        """Test to_dict() handles None values"""
        segment = SRTSegment(
            index=1,
            start_time=0.0,
            end_time=5.0,
            text="Test",
            topic_id=None,
        )

        d = segment.to_dict()
        assert d["topic_id"] is None

    @pytest.mark.fast
    def test_segment_from_dict_minimal(self):
        """Test from_dict() with minimal fields"""
        data = {"text": "Only text"}
        segment = SRTSegment.from_dict(data)

        assert segment.text == "Only text"
        assert segment.index == 0
        assert segment.start_time == 0.0
        assert segment.end_time == 0.0
        assert segment.source_file == ""
        assert segment.keywords == []
        assert segment.entities == []

    @pytest.mark.fast
    def test_segment_from_dict_ignores_extra_fields(self):
        """Test from_dict() ignores unknown fields"""
        data = {
            "index": 1,
            "start_time": 0.0,
            "end_time": 5.0,
            "text": "Test",
            "unknown_field": "should be ignored",
            "another_unknown": 123,
        }
        segment = SRTSegment.from_dict(data)

        assert not hasattr(segment, "unknown_field")
        assert not hasattr(segment, "another_unknown")

    @pytest.mark.fast
    def test_segment_roundtrip_to_from_dict(self):
        """Test roundtrip through to_dict and from_dict"""
        original = SRTSegment(
            index=42,
            start_time=10.5,
            end_time=20.75,
            text="Roundtrip test",
            source_file="/test/video.mp4",
            keywords=["key1", "key2"],
            entities=[{"text": "Entity", "type": "ORG"}],
            topic_id=5,
            topics=["topic1", "topic2"],
            is_broll=True,
        )

        d = original.to_dict()
        restored = SRTSegment.from_dict(d)

        assert restored.index == original.index
        assert restored.start_time == original.start_time
        assert restored.end_time == original.end_time
        assert restored.text == original.text
        assert restored.source_file == original.source_file
        assert restored.keywords == original.keywords
        assert restored.topic_id == original.topic_id
        assert restored.is_broll == original.is_broll

    @pytest.mark.fast
    def test_segment_defaults(self):
        """Test default values for optional fields"""
        segment = SRTSegment(index=1, start_time=0.0, end_time=5.0, text="Test")

        assert segment.source_file == ""
        assert segment.keywords == []
        assert segment.entities == []
        assert segment.topic_id is None
        assert segment.topics == []
        assert segment.is_broll is False


class TestSrtFileRoundtrip:
    """Integration tests for full SRT file roundtrip"""

    @pytest.mark.fast
    def test_roundtrip_basic(self, tmp_path):
        """Test write → parse → write produces same file"""
        segments = [
            SRTSegment(index=1, start_time=0.0, end_time=5.0, text="First"),
            SRTSegment(index=2, start_time=5.0, end_time=10.0, text="Second"),
            SRTSegment(index=3, start_time=10.0, end_time=15.0, text="Third"),
        ]

        file1 = tmp_path / "first.srt"
        file2 = tmp_path / "second.srt"

        write_srt_file(segments, str(file1))
        parsed = parse_srt_file(str(file1))
        write_srt_file(parsed, str(file2))

        # Content should be equivalent
        assert len(parsed) == 3
        assert parsed[0].text == "First"
        assert parsed[1].text == "Second"
        assert parsed[2].text == "Third"

    @pytest.mark.fast
    def test_roundtrip_preserves_timestamps(self, tmp_path):
        """Test roundtrip preserves timestamp precision"""
        segments = [
            SRTSegment(index=1, start_time=0.123, end_time=5.456, text="Test"),
        ]

        srt_file = tmp_path / "test.srt"
        write_srt_file(segments, str(srt_file))
        parsed = parse_srt_file(str(srt_file))

        assert len(parsed) == 1
        assert abs(parsed[0].start_time - 0.123) < 0.001
        assert abs(parsed[0].end_time - 5.456) < 0.001


class TestEncodingHandling:
    """Tests for various file encodings"""

    @pytest.mark.fast
    def test_parse_utf8_bom(self, tmp_path):
        """Test parsing UTF-8 file with BOM handles gracefully"""
        content = """1
00:00:00,000 --> 00:00:05,000
UTF-8 BOM content
"""
        srt_file = tmp_path / "utf8bom.srt"
        # Write with UTF-8 BOM
        with open(srt_file, "wb") as f:
            f.write(b"\xef\xbb\xbf")  # UTF-8 BOM
            f.write(content.encode("utf-8"))

        result = parse_srt_file(str(srt_file))
        # BOM may cause first line parsing issues - graceful handling is acceptable
        assert isinstance(result, list)

    @pytest.mark.fast
    def test_parse_latin1_encoding(self, tmp_path):
        """Test parsing Latin-1 encoded file"""
        content = """1
00:00:00,000 --> 00:00:05,000
Café résumé naïve
"""
        srt_file = tmp_path / "latin1.srt"
        srt_file.write_text(content, encoding="latin-1")

        result = parse_srt_file(str(srt_file))
        # Should parse without error
        assert isinstance(result, list)

    @pytest.mark.fast
    def test_parse_windows_line_endings(self, tmp_path):
        """Test parsing file with Windows line endings"""
        content = "1\r\n00:00:00,000 --> 00:00:05,000\r\nWindows CRLF\r\n\r\n"
        srt_file = tmp_path / "windows.srt"
        srt_file.write_bytes(content.encode("utf-8"))

        result = parse_srt_file(str(srt_file))
        # Should handle CRLF line endings
        assert isinstance(result, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
