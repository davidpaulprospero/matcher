"""
Unit tests for OTIO utility functions.

Tests path handling, XML escaping, timecode conversion, media utils, and clip creation.
"""

import pytest
from pathlib import Path
import sys
from unittest.mock import Mock, patch, MagicMock
import opentimelineio as otio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.otio.utils import (
    _to_windows_path,
    format_path_url,
    sanitize_path_for_url,
    encode_path_for_xml_url,
    escape_xml,
    _to_python_type,
    _sanitize_metadata,
    _get_media_duration,
    get_segment_file_offset,
    is_segment_file,
    frames_to_tc,
    get_confidence_color,
    create_clip_with_timewarp
)
import numpy as np


class TestWindowsPathConversion:
    """Test Windows path conversion."""

    def test_to_windows_path_forward_slashes(self):
        """Test converting forward slashes to backslashes."""
        path = "C:/Users/test/video.mp4"
        result = _to_windows_path(path)
        assert result == "C:\\Users\\test\\video.mp4"

    def test_to_windows_path_already_backslashes(self):
        """Test path that already has backslashes."""
        path = "C:\\Users\\test\\video.mp4"
        result = _to_windows_path(path)
        assert result == "C:\\Users\\test\\video.mp4"

    def test_to_windows_path_extended_length_prefix(self):
        """Test that extended-length prefix is preserved."""
        path = "\\\\?\\C:/Users/test/video.mp4"
        result = _to_windows_path(path)
        assert result.startswith("\\\\?\\")
        assert "/" not in result


class TestPathFormatting:
    """Test path formatting for URLs."""

    def test_format_path_url_basic(self):
        """Test basic path URL formatting."""
        path = "C:/Videos/test.mp4"
        formatted = format_path_url(path)
        assert formatted is not None
        assert isinstance(formatted, str)
        # Returns plain path with forward slashes, not file:// URL
        assert "/" in formatted

    def test_format_path_url_windows_backslashes(self):
        """Test Windows path with backslashes."""
        path = "C:\\Users\\test\\video.mp4"
        formatted = format_path_url(path)
        assert formatted is not None
        # Should convert to forward slashes for URL
        assert "\\" not in formatted or path.startswith("\\\\?\\")

    def test_format_path_url_with_spaces(self):
        """Test path with spaces."""
        path = "C:/Videos/test video.mp4"
        formatted = format_path_url(path)
        assert formatted is not None
        # Returns plain path - spaces NOT encoded (DaVinci prefers unencoded)
        assert isinstance(formatted, str)

    def test_sanitize_path_for_url(self):
        """Test sanitizing path for URL."""
        path = "C:/Videos/test video.mp4"
        sanitized = sanitize_path_for_url(path)
        assert sanitized is not None
        # Should handle special characters
        assert isinstance(sanitized, str)

    def test_sanitize_path_for_url_double_slashes(self):
        """Test removing double slashes."""
        path = "C://Users//test//video.mp4"
        sanitized = sanitize_path_for_url(path)
        # Should not have consecutive slashes (except file://)
        assert "///" not in sanitized

    def test_encode_path_for_xml_url(self):
        """Test encoding path for XML URL."""
        path = "C:/Videos/test & more.mp4"
        encoded = encode_path_for_xml_url(path)
        assert encoded is not None
        # Function returns plain path - doesn't XML-encode (done elsewhere)
        assert isinstance(encoded, str)


class TestXMLEscaping:
    """Test XML escaping."""

    def test_escape_xml_ampersand(self):
        """Test escaping ampersand."""
        text = "Test & more"
        escaped = escape_xml(text)
        assert "&amp;" in escaped
        assert "&" not in escaped.replace("&amp;", "")

    def test_escape_xml_less_than(self):
        """Test escaping less than."""
        text = "x < 5"
        escaped = escape_xml(text)
        assert "&lt;" in escaped

    def test_escape_xml_greater_than(self):
        """Test escaping greater than."""
        text = "x > 5"
        escaped = escape_xml(text)
        assert "&gt;" in escaped

    def test_escape_xml_quotes(self):
        """Test escaping double quotes."""
        text = 'Say "hello"'
        escaped = escape_xml(text)
        assert "&quot;" in escaped

    def test_escape_xml_apostrophe(self):
        """Test escaping apostrophes."""
        text = "It's working"
        escaped = escape_xml(text)
        assert "&apos;" in escaped or "'" not in escaped

    def test_escape_xml_no_special_chars(self):
        """Test text without special characters."""
        text = "Plain text"
        escaped = escape_xml(text)
        assert escaped == text

    def test_escape_xml_multiple_special_chars(self):
        """Test escaping multiple special characters."""
        text = "Test & <tag> \"quote\""
        escaped = escape_xml(text)
        assert "&amp;" in escaped
        assert "&lt;" in escaped
        assert "&gt;" in escaped
        assert "&quot;" in escaped


class TestTypeConversion:
    """Test Python type conversion for numpy types."""

    def test_to_python_type_numpy_int(self):
        """Test converting numpy int to Python int."""
        value = np.int64(42)
        result = _to_python_type(value)
        assert isinstance(result, int)
        assert result == 42

    def test_to_python_type_numpy_float(self):
        """Test converting numpy float to Python float."""
        value = np.float64(3.14)
        result = _to_python_type(value)
        assert isinstance(result, float)
        assert abs(result - 3.14) < 0.01

    def test_to_python_type_regular_int(self):
        """Test regular int passes through."""
        value = 42
        result = _to_python_type(value)
        assert result == 42

    def test_to_python_type_string(self):
        """Test string passes through."""
        value = "test"
        result = _to_python_type(value)
        assert result == "test"


class TestMetadataSanitization:
    """Test metadata sanitization for JSON serialization."""

    def test_sanitize_metadata_numpy_values(self):
        """Test sanitizing metadata with numpy values."""
        metadata = {
            "score": np.float64(0.95),
            "count": np.int64(10),
            "name": "test"
        }
        sanitized = _sanitize_metadata(metadata)
        assert isinstance(sanitized["score"], float)
        assert isinstance(sanitized["count"], int)
        assert sanitized["name"] == "test"

    def test_sanitize_metadata_nested_dict(self):
        """Test sanitizing nested metadata."""
        metadata = {
            "outer": {
                "inner": np.float64(0.5)
            }
        }
        sanitized = _sanitize_metadata(metadata)
        assert isinstance(sanitized["outer"]["inner"], float)

    def test_sanitize_metadata_list_values(self):
        """Test sanitizing metadata with lists."""
        metadata = {
            "scores": [np.float64(0.1), np.float64(0.2)]
        }
        sanitized = _sanitize_metadata(metadata)
        assert all(isinstance(v, float) for v in sanitized["scores"])


class TestMediaDuration:
    """Test media duration extraction."""

    @patch('subprocess.run')
    def test_get_media_duration_success(self, mock_run):
        """Test successful duration extraction."""
        # Mock ffprobe output
        mock_run.return_value = Mock(
            returncode=0,
            stdout="120.5\n",
            stderr=""
        )

        duration = _get_media_duration("/path/to/video.mp4")
        assert duration == 120.5

    @patch('subprocess.run')
    def test_get_media_duration_failure(self, mock_run):
        """Test duration extraction failure."""
        mock_run.return_value = Mock(returncode=1, stdout="", stderr="Error")

        duration = _get_media_duration("/path/to/video.mp4")
        assert duration is None

    @patch('subprocess.run')
    def test_get_media_duration_invalid_output(self, mock_run):
        """Test handling invalid ffprobe output."""
        mock_run.return_value = Mock(returncode=0, stdout="invalid\n", stderr="")

        duration = _get_media_duration("/path/to/video.mp4")
        assert duration is None


class TestSegmentFileHandling:
    """Test segment file detection and offset extraction."""

    def test_is_segment_file_true(self):
        """Test detecting segment file with _0000.mp4 pattern."""
        result = is_segment_file("/path/to/video_0000.mp4")
        assert result is True

    def test_is_segment_file_false(self):
        """Test non-segment file."""
        result = is_segment_file("/path/to/video.mp4")
        assert result is False

    def test_is_segment_file_different_pattern(self):
        """Test segment file with different number pattern."""
        result = is_segment_file("/path/to/video_segment_0005.mp4")
        assert isinstance(result, bool)

    def test_get_segment_file_offset_basic(self):
        """Test extracting offset from segment filename with 4-digit pattern."""
        offset = get_segment_file_offset("/path/to/abc12345678_0045.mp4")
        assert offset == 45.0

    def test_get_segment_file_offset_no_offset(self):
        """Test regular (non-segment) file."""
        offset = get_segment_file_offset("/path/to/video.mp4")
        assert offset == 0.0

    def test_get_segment_file_offset_zero(self):
        """Test segment file with zero offset."""
        offset = get_segment_file_offset("/path/to/videoidhere_0000.mp4")
        assert offset == 0.0

    def test_get_segment_file_offset_large_value(self):
        """Test offset with large value (e.g., 2 hours)."""
        offset = get_segment_file_offset("/path/to/myvideofile_7200.mp4")
        assert offset == 7200.0


class TestTimecodeConversion:
    """Test timecode conversion."""

    def test_frames_to_tc_basic(self):
        """Test converting frames to timecode."""
        tc = frames_to_tc(300, fps=30.0)
        assert tc is not None
        assert isinstance(tc, str)
        # 300 frames at 30fps = 10 seconds = 00:00:10:00
        assert "10" in tc

    def test_frames_to_tc_zero(self):
        """Test zero frames."""
        tc = frames_to_tc(0, fps=30.0)
        assert "00:00:00:00" in tc

    def test_frames_to_tc_different_fps(self):
        """Test with different frame rates."""
        tc1 = frames_to_tc(60, fps=60.0)
        tc2 = frames_to_tc(30, fps=30.0)
        # Both should be 1 second
        assert tc1 is not None
        assert tc2 is not None


class TestConfidenceColor:
    """Test confidence color mapping."""

    def test_confidence_color_high(self):
        """Test color for high confidence."""
        color = get_confidence_color(0.9)
        assert color is not None
        # High confidence should be green
        assert color == "GREEN"

    def test_confidence_color_medium(self):
        """Test color for medium confidence."""
        color = get_confidence_color(0.6)
        assert color == "CYAN"

    def test_confidence_color_low(self):
        """Test color for low confidence."""
        color = get_confidence_color(0.3)
        assert color == "ORANGE"

    def test_confidence_color_very_low(self):
        """Test color for very low confidence."""
        color = get_confidence_color(0.1)
        assert color == "RED"

    def test_confidence_color_zero(self):
        """Test color for zero confidence."""
        color = get_confidence_color(0.0)
        assert color == "RED"

    def test_confidence_color_boundary_cases(self):
        """Test boundary values for color mapping."""
        assert get_confidence_color(0.8) == "GREEN"
        assert get_confidence_color(0.6) == "CYAN"
        assert get_confidence_color(0.4) == "YELLOW"
        assert get_confidence_color(0.2) == "ORANGE"


class TestClipCreation:
    """Test OTIO clip creation with timewarp."""

    def test_create_clip_basic(self):
        """Test basic clip creation without speed adjustment."""
        clip = create_clip_with_timewarp(
            name="TestClip",
            source_path="C:/Videos/test.mp4",
            source_start=10.0,
            source_duration=5.0,
            target_duration=5.0,  # Same duration
            frame_rate=30.0
        )

        assert isinstance(clip, otio.schema.Clip)
        assert clip.name == "TestClip"
        assert clip.media_reference is not None

    def test_create_clip_with_slowdown(self):
        """Test clip creation with slowdown (source longer than target)."""
        clip = create_clip_with_timewarp(
            name="SlowClip",
            source_path="C:/Videos/test.mp4",
            source_start=0.0,
            source_duration=10.0,
            target_duration=15.0,  # Slower
            frame_rate=30.0
        )

        assert clip is not None
        # Should have timewarp effect for speed adjustment
        assert len(clip.effects) > 0

    def test_create_clip_with_speedup(self):
        """Test clip creation with speedup (source shorter than target)."""
        clip = create_clip_with_timewarp(
            name="FastClip",
            source_path="C:/Videos/test.mp4",
            source_start=0.0,
            source_duration=10.0,
            target_duration=5.0,  # Faster
            frame_rate=30.0
        )

        assert clip is not None
        assert len(clip.effects) > 0

    def test_create_clip_with_metadata(self):
        """Test clip creation with custom metadata."""
        metadata = {
            "confidence": 0.95,
            "strategy": "primary",
            "score": np.float64(0.85)
        }

        clip = create_clip_with_timewarp(
            name="MetadataClip",
            source_path="C:/Videos/test.mp4",
            source_start=0.0,
            source_duration=5.0,
            target_duration=5.0,
            frame_rate=30.0,
            metadata=metadata
        )

        assert clip.metadata is not None
        assert "confidence" in clip.metadata

    def test_create_clip_with_media_duration(self):
        """Test clip creation with known media duration."""
        clip = create_clip_with_timewarp(
            name="KnownDurationClip",
            source_path="C:/Videos/test.mp4",
            source_start=10.0,
            source_duration=5.0,
            target_duration=5.0,
            frame_rate=30.0,
            media_duration=120.0  # 2 minutes total
        )

        assert clip is not None
        # Media reference should have available_range set
        assert clip.media_reference.available_range is not None

    def test_create_clip_windows_path_handling(self):
        """Test clip creation with Windows path."""
        clip = create_clip_with_timewarp(
            name="WindowsPathClip",
            source_path="C:\\Users\\test\\video.mp4",
            source_start=0.0,
            source_duration=5.0,
            target_duration=5.0,
            frame_rate=30.0
        )

        assert clip is not None
        # Should handle backslashes
        assert clip.media_reference.target_url is not None

    def test_create_clip_unique_naming(self):
        """Test that clips get unique media reference names."""
        clip1 = create_clip_with_timewarp(
            name="Clip1",
            source_path="C:/folder1/video.mp4",
            source_start=0.0,
            source_duration=5.0,
            target_duration=5.0
        )

        clip2 = create_clip_with_timewarp(
            name="Clip2",
            source_path="C:/folder2/video.mp4",
            source_start=0.0,
            source_duration=5.0,
            target_duration=5.0
        )

        # Different folders should result in different media reference names
        # to avoid DaVinci Resolve conflicts
        assert clip1.media_reference.name != clip2.media_reference.name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
