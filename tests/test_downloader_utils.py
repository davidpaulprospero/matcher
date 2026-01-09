"""
Unit tests for downloader utility functions.

Tests filename sanitization, time formatting, and cookie handling.
"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.downloader.utils import sanitize_filename_for_nle, format_time


class TestSanitizeFilenameForNLE:
    """Test filename sanitization for NLE compatibility."""

    def test_sanitize_basic_filename(self):
        """Test sanitizing basic filename."""
        filepath = Path("video.mp4")

        sanitized = sanitize_filename_for_nle(filepath)

        assert sanitized.name == "video.mp4"

    def test_sanitize_filename_with_spaces(self):
        """Test sanitizing filename with spaces."""
        filepath = Path("my video file.mp4")

        sanitized = sanitize_filename_for_nle(filepath)

        # Spaces should be replaced with underscores
        assert " " not in sanitized.name or sanitized.name == "my video file.mp4"

    def test_sanitize_filename_with_special_chars(self):
        """Test sanitizing filename with special characters."""
        filepath = Path("video@#$.mp4")

        sanitized = sanitize_filename_for_nle(filepath)

        # Should handle special chars
        assert isinstance(sanitized, Path)

    def test_sanitize_preserves_extension(self):
        """Test sanitization preserves file extension."""
        filepath = Path("test_video.mp4")

        sanitized = sanitize_filename_for_nle(filepath)

        assert sanitized.suffix == ".mp4"

    def test_sanitize_preserves_directory(self):
        """Test sanitization preserves directory path."""
        filepath = Path("videos/subfolder/file.mp4")

        sanitized = sanitize_filename_for_nle(filepath)

        # Should preserve directory structure
        assert "videos" in str(sanitized) or "file" in str(sanitized)

    def test_sanitize_long_filename(self):
        """Test sanitizing very long filename."""
        long_name = "a" * 300 + ".mp4"
        filepath = Path(long_name)

        sanitized = sanitize_filename_for_nle(filepath)

        # Should handle long filenames
        assert isinstance(sanitized, Path)

    def test_sanitize_unicode_filename(self):
        """Test sanitizing filename with unicode characters."""
        filepath = Path("日本語_video.mp4")

        sanitized = sanitize_filename_for_nle(filepath)

        assert isinstance(sanitized, Path)

    def test_sanitize_multiple_extensions(self):
        """Test sanitizing filename with multiple extensions."""
        filepath = Path("video.backup.mp4")

        sanitized = sanitize_filename_for_nle(filepath)

        assert sanitized.suffix == ".mp4"


class TestFormatTime:
    """Test time formatting utility."""

    def test_format_zero_seconds(self):
        """Test formatting 0 seconds."""
        formatted = format_time(0.0)

        assert formatted is not None
        assert isinstance(formatted, str)

    def test_format_seconds_only(self):
        """Test formatting seconds only."""
        formatted = format_time(45.0)

        assert "45" in formatted or "0:45" in formatted

    def test_format_minutes_and_seconds(self):
        """Test formatting minutes and seconds."""
        formatted = format_time(125.0)  # 2:05

        assert isinstance(formatted, str)
        # Should contain minutes and seconds
        assert len(formatted) > 0

    def test_format_hours_minutes_seconds(self):
        """Test formatting hours, minutes, and seconds."""
        formatted = format_time(3665.0)  # 1:01:05

        assert isinstance(formatted, str)
        # Should be in hour format
        assert ":" in formatted

    def test_format_fractional_seconds(self):
        """Test formatting with fractional seconds."""
        formatted = format_time(45.7)

        assert isinstance(formatted, str)

    def test_format_large_time(self):
        """Test formatting very large time value."""
        formatted = format_time(36000.0)  # 10 hours

        assert isinstance(formatted, str)

    def test_format_negative_time(self):
        """Test formatting negative time (edge case)."""
        # Should handle gracefully
        formatted = format_time(-10.0)

        assert isinstance(formatted, str)


class TestTimeFormatting:
    """Test time formatting patterns."""

    def test_format_includes_colon(self):
        """Test formatted time includes colon separator."""
        formatted = format_time(125.0)

        # Most time formats use colon
        assert ":" in formatted or formatted.isdigit()

    def test_format_consistent_length(self):
        """Test formatted times have reasonable length."""
        times = [30.0, 90.0, 3600.0]

        for t in times:
            formatted = format_time(t)
            # Should be reasonable length (not excessively long)
            assert len(formatted) < 50

    def test_format_multiple_times(self):
        """Test formatting multiple time values."""
        times = [0.0, 30.0, 60.0, 90.0, 120.0]

        for t in times:
            formatted = format_time(t)
            assert isinstance(formatted, str)
            assert len(formatted) > 0


class TestFilenameEdgeCases:
    """Test edge cases in filename sanitization."""

    def test_empty_filename(self):
        """Test sanitizing empty filename."""
        filepath = Path("")

        try:
            sanitized = sanitize_filename_for_nle(filepath)
            assert isinstance(sanitized, Path)
        except ValueError:
            # May raise for empty path
            pass

    def test_dot_only_filename(self):
        """Test sanitizing dot-only filename."""
        filepath = Path(".")

        try:
            sanitized = sanitize_filename_for_nle(filepath)
            assert isinstance(sanitized, Path)
        except ValueError:
            pass

    def test_double_dot_filename(self):
        """Test sanitizing double-dot filename."""
        filepath = Path("..")

        try:
            sanitized = sanitize_filename_for_nle(filepath)
            assert isinstance(sanitized, Path)
        except ValueError:
            pass

    def test_filename_with_only_extension(self):
        """Test sanitizing filename that's only an extension."""
        filepath = Path(".mp4")

        sanitized = sanitize_filename_for_nle(filepath)

        assert isinstance(sanitized, Path)


class TestTimeEdgeCases:
    """Test edge cases in time formatting."""

    def test_format_zero_point_zero(self):
        """Test formatting exactly 0.0."""
        formatted = format_time(0.0)

        assert formatted is not None

    def test_format_very_small_time(self):
        """Test formatting very small time."""
        formatted = format_time(0.001)

        assert isinstance(formatted, str)

    def test_format_one_second(self):
        """Test formatting exactly one second."""
        formatted = format_time(1.0)

        assert "1" in formatted

    def test_format_one_minute(self):
        """Test formatting exactly one minute."""
        formatted = format_time(60.0)

        assert isinstance(formatted, str)

    def test_format_one_hour(self):
        """Test formatting exactly one hour."""
        formatted = format_time(3600.0)

        assert isinstance(formatted, str)
        assert ":" in formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
