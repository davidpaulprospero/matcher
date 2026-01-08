"""
Unit tests for OTIO utility functions.

Tests path handling, XML escaping, and timecode conversion.
"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.otio.utils import (
    format_path_url,
    sanitize_path_for_url,
    escape_xml,
    frames_to_tc,
    get_confidence_color,
    is_segment_file
)


class TestPathFormatting:
    """Test path formatting for URLs."""

    def test_format_path_url_basic(self):
        """Test basic path URL formatting."""
        path = "C:/Videos/test.mp4"
        formatted = format_path_url(path)
        assert formatted is not None
        assert isinstance(formatted, str)

    def test_sanitize_path_for_url(self):
        """Test sanitizing path for URL."""
        path = "C:/Videos/test video.mp4"
        sanitized = sanitize_path_for_url(path)
        assert sanitized is not None


class TestXMLEscaping:
    """Test XML escaping."""

    def test_escape_xml_basic(self):
        """Test escaping basic XML characters."""
        text = "Test & more"
        escaped = escape_xml(text)
        assert "&amp;" in escaped

    def test_escape_xml_no_special_chars(self):
        """Test text without special characters."""
        text = "Plain text"
        escaped = escape_xml(text)
        assert escaped == text


class TestTimecodeConversion:
    """Test timecode conversion."""

    def test_frames_to_tc_basic(self):
        """Test converting frames to timecode."""
        tc = frames_to_tc(300, fps=30.0)
        assert tc is not None
        assert isinstance(tc, str)


class TestConfidenceColor:
    """Test confidence color mapping."""

    def test_confidence_color_high(self):
        """Test color for high confidence."""
        color = get_confidence_color(0.9)
        assert color is not None


class TestSegmentFile:
    """Test segment file detection."""

    def test_is_segment_file(self):
        """Test detecting segment file."""
        result = is_segment_file("/path/to/video_seg_00001.mp4")
        assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
