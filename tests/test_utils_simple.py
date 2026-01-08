"""
Unit tests for utils module - simplified version.

Tests only functions that exist in the actual utils.py module.
"""

import pytest
from pathlib import Path
import sys
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    SRTSegment,
    is_embeddings_empty,
    normalize_path,
    sanitize_path,
    parse_srt_timestamp,
    format_srt_timestamp,
    parse_srt_file,
    write_srt_file
)


class TestSRTSegment:
    """Test SRTSegment dataclass."""

    def test_create_srt_segment(self):
        """Test creating SRT segment."""
        segment = SRTSegment(
            index=1,
            start_time=0.0,
            end_time=5.0,
            text="Test subtitle text"
        )

        assert segment.index == 1
        assert segment.start_time == 0.0
        assert segment.end_time == 5.0
        assert segment.text == "Test subtitle text"

    def test_srt_segment_duration(self):
        """Test computing segment duration."""
        segment = SRTSegment(
            index=1,
            start_time=10.0,
            end_time=25.5,
            text="Test"
        )

        duration = segment.end_time - segment.start_time
        assert duration == 15.5

    def test_srt_segment_with_source_file(self):
        """Test segment with source file."""
        segment = SRTSegment(
            index=1,
            start_time=0.0,
            end_time=5.0,
            text="Test",
            source_file="/path/to/video.mp4"
        )

        assert segment.source_file == "/path/to/video.mp4"

    def test_srt_segment_equality(self):
        """Test comparing SRT segments."""
        seg1 = SRTSegment(index=1, start_time=0.0, end_time=5.0, text="Test")
        seg2 = SRTSegment(index=1, start_time=0.0, end_time=5.0, text="Test")
        seg3 = SRTSegment(index=2, start_time=5.0, end_time=10.0, text="Different")

        assert seg1 == seg2
        assert seg1 != seg3


class TestSRTTimestamps:
    """Test SRT timestamp formatting and parsing."""

    def test_format_srt_timestamp_basic(self):
        """Test basic timestamp formatting."""
        formatted = format_srt_timestamp(90.5)

        # Should be in format: HH:MM:SS,mmm
        assert ":" in formatted
        assert "," in formatted

    def test_format_srt_timestamp_hours(self):
        """Test formatting with hours."""
        formatted = format_srt_timestamp(3665.0)  # 1 hour, 1 minute, 5 seconds

        assert formatted.count(":") == 2

    def test_parse_srt_timestamp_basic(self):
        """Test parsing SRT timestamp."""
        seconds = parse_srt_timestamp("00:01:30,000")

        assert seconds == 90.0

    def test_parse_srt_timestamp_with_milliseconds(self):
        """Test parsing with milliseconds."""
        seconds = parse_srt_timestamp("00:00:01,500")

        assert seconds == 1.5

    def test_format_parse_roundtrip(self):
        """Test formatting and parsing round-trip."""
        original = 125.75

        formatted = format_srt_timestamp(original)
        parsed = parse_srt_timestamp(formatted)

        assert abs(parsed - original) < 0.01  # Within 10ms


class TestPathHandling:
    """Test path normalization and handling."""

    def test_normalize_path_forward_slash(self):
        """Test normalizing path with forward slashes."""
        path = "C:/Users/Test/video.mp4"

        normalized = normalize_path(path)

        assert normalized is not None

    def test_normalize_path_backslash(self):
        """Test normalizing path with backslashes."""
        path = "C:\\Users\\Test\\video.mp4"

        normalized = normalize_path(path)

        assert normalized is not None

    def test_sanitize_path_basic(self):
        """Test sanitizing path."""
        path = "C:/Users/Test/video.mp4"

        sanitized = sanitize_path(path)

        assert sanitized is not None

    def test_sanitize_path_special_chars(self):
        """Test sanitizing path with special characters."""
        path = "C:/Users/Test: Video/file.mp4"

        sanitized = sanitize_path(path)

        # Should handle special characters
        assert sanitized is not None


class TestSRTFileParsing:
    """Test parsing SRT files."""

    def test_parse_srt_basic(self, tmp_path):
        """Test parsing basic SRT file."""
        srt_content = """1
00:00:00,000 --> 00:00:05,000
First subtitle

2
00:00:05,000 --> 00:00:10,000
Second subtitle
"""
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(srt_content)

        segments = parse_srt_file(str(srt_file))

        assert len(segments) == 2
        assert segments[0].text == "First subtitle"
        assert segments[1].text == "Second subtitle"

    def test_parse_srt_with_timestamps(self, tmp_path):
        """Test parsing SRT preserves timestamps."""
        srt_content = """1
00:00:10,500 --> 00:00:15,750
Test subtitle
"""
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(srt_content)

        segments = parse_srt_file(str(srt_file))

        assert len(segments) == 1
        assert segments[0].start_time == 10.5
        assert segments[0].end_time == 15.75

    def test_write_srt_basic(self, tmp_path):
        """Test writing SRT file."""
        segments = [
            SRTSegment(
                index=1,
                start_time=0.0,
                end_time=5.0,
                text="First subtitle"
            ),
            SRTSegment(
                index=2,
                start_time=5.0,
                end_time=10.0,
                text="Second subtitle"
            )
        ]

        output_file = tmp_path / "output.srt"
        write_srt_file(segments, str(output_file))

        assert output_file.exists()

        # Verify content
        content = output_file.read_text()
        assert "First subtitle" in content
        assert "Second subtitle" in content

    def test_write_read_roundtrip(self, tmp_path):
        """Test writing and reading SRT round-trip."""
        original_segments = [
            SRTSegment(
                index=1,
                start_time=0.0,
                end_time=5.0,
                text="Test 1"
            ),
            SRTSegment(
                index=2,
                start_time=5.0,
                end_time=10.0,
                text="Test 2"
            )
        ]

        srt_file = tmp_path / "test.srt"
        write_srt_file(original_segments, str(srt_file))

        # Read back
        loaded_segments = parse_srt_file(str(srt_file))

        assert len(loaded_segments) == len(original_segments)
        for orig, loaded in zip(original_segments, loaded_segments):
            assert orig.text == loaded.text
            assert abs(orig.start_time - loaded.start_time) < 0.01
            assert abs(orig.end_time - loaded.end_time) < 0.01


class TestEmbeddingHelpers:
    """Test embedding helper functions."""

    def test_is_embeddings_empty_none(self):
        """Test checking None embeddings."""
        assert is_embeddings_empty(None)

    def test_is_embeddings_empty_array(self):
        """Test checking empty array."""
        empty_array = np.array([])

        assert is_embeddings_empty(empty_array)

    def test_is_embeddings_not_empty(self):
        """Test checking non-empty embeddings."""
        embeddings = np.random.randn(10, 1024).astype(np.float32)

        assert not is_embeddings_empty(embeddings)

    def test_is_embeddings_single_vector(self):
        """Test checking single vector."""
        single_vec = np.random.randn(1024).astype(np.float32)

        assert not is_embeddings_empty(single_vec)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
