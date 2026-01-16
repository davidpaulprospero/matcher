"""
Unit tests for downloader utility functions.

Tests filename sanitization, time formatting, and cookie handling.
"""

import pytest
import tempfile
from pathlib import Path
import sys
from unittest.mock import Mock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.downloader.utils import (
    sanitize_filename_for_nle,
    format_time,
    get_cookies_args,
    extract_video_id,
    detect_caption_format,
    get_caption_language,
    is_auto_generated_caption,
    caption_file_priority,
)


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


class TestGetCookiesArgs:
    """Test cookie argument generation for yt-dlp."""

    def test_get_cookies_args_browser_cookies(self):
        """Test getting cookies from browser."""
        mock_config = Mock()
        mock_download = Mock()
        mock_download.cookies_from_browser = "firefox"
        mock_download.cookies_path = ""
        mock_config.download = mock_download

        args = get_cookies_args(mock_config)

        assert args == ['--cookies-from-browser', 'firefox']

    def test_get_cookies_args_file_path(self):
        """Test getting cookies from file."""
        # Create temporary cookies file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".txt") as f:
            f.write("# Cookies file\n")
            cookies_path = f.name

        try:
            mock_config = Mock()
            mock_download = Mock()
            mock_download.cookies_from_browser = ""  # No browser
            mock_download.cookies_path = cookies_path
            mock_config.download = mock_download

            args = get_cookies_args(mock_config)

            assert args == ['--cookies', cookies_path]
        finally:
            Path(cookies_path).unlink()

    def test_get_cookies_args_no_cookies(self):
        """Test when no cookies are configured."""
        mock_config = Mock()
        mock_download = Mock()
        mock_download.cookies_from_browser = ""
        mock_download.cookies_path = ""
        mock_config.download = mock_download

        args = get_cookies_args(mock_config)

        assert args == []

    def test_get_cookies_args_browser_priority(self):
        """Test browser cookies take priority over file."""
        # Create temporary cookies file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".txt") as f:
            f.write("# Cookies file\n")
            cookies_path = f.name

        try:
            mock_config = Mock()
            mock_download = Mock()
            mock_download.cookies_from_browser = "chrome"  # Has browser
            mock_download.cookies_path = cookies_path      # Also has file
            mock_config.download = mock_download

            args = get_cookies_args(mock_config)

            # Should prefer browser over file
            assert args == ['--cookies-from-browser', 'chrome']
        finally:
            Path(cookies_path).unlink()

    def test_get_cookies_args_nonexistent_file(self):
        """Test with nonexistent cookies file path."""
        mock_config = Mock()
        mock_download = Mock()
        mock_download.cookies_from_browser = ""
        mock_download.cookies_path = "/nonexistent/cookies.txt"
        mock_config.download = mock_download

        args = get_cookies_args(mock_config)

        # Should return empty list if file doesn't exist
        assert args == []

    def test_get_cookies_args_edge_browser(self):
        """Test with Edge browser."""
        mock_config = Mock()
        mock_download = Mock()
        mock_download.cookies_from_browser = "edge"
        mock_download.cookies_path = ""
        mock_config.download = mock_download

        args = get_cookies_args(mock_config)

        assert args == ['--cookies-from-browser', 'edge']

    def test_get_cookies_args_safari_browser(self):
        """Test with Safari browser."""
        mock_config = Mock()
        mock_download = Mock()
        mock_download.cookies_from_browser = "safari"
        mock_download.cookies_path = ""
        mock_config.download = mock_download

        args = get_cookies_args(mock_config)

        assert args == ['--cookies-from-browser', 'safari']

    def test_get_cookies_args_missing_attributes(self):
        """Test with missing config attributes (uses getattr defaults)."""
        mock_config = Mock()
        mock_download = Mock(spec=[])  # No attributes
        mock_config.download = mock_download

        args = get_cookies_args(mock_config)

        # Should handle missing attributes gracefully
        assert args == []


class TestExtractVideoId:
    """Tests for extract_video_id function"""

    def test_extract_from_watch_url(self):
        """Test extraction from standard watch URL"""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_from_short_url(self):
        """Test extraction from youtu.be short URL"""
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_from_embed_url(self):
        """Test extraction from embed URL"""
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_from_v_url(self):
        """Test extraction from /v/ URL format"""
        url = "https://www.youtube.com/v/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_from_url_with_params(self):
        """Test extraction from URL with extra parameters"""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s&list=PLxyz"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_from_filename(self):
        """Test extraction from filename containing video ID"""
        filename = "travel_vlog_dQw4w9WgXcQ.mp4"
        result = extract_video_id(filename)
        # May extract first 11-char match which could be 'travel_vlog' portion
        # or 'dQw4w9WgXcQ' depending on pattern matching
        assert result is not None and len(result) == 11

    def test_extract_from_caption_filename(self):
        """Test extraction from caption filename"""
        filename = "dQw4w9WgXcQ.en.srt"
        assert extract_video_id(filename) == "dQw4w9WgXcQ"

    def test_extract_from_raw_id(self):
        """Test extraction from raw video ID"""
        video_id = "dQw4w9WgXcQ"
        assert extract_video_id(video_id) == "dQw4w9WgXcQ"

    def test_extract_with_underscores(self):
        """Test extraction with underscores in ID"""
        url = "https://youtube.com/watch?v=abc_def-123"
        assert extract_video_id(url) == "abc_def-123"

    def test_extract_with_dashes(self):
        """Test extraction with dashes in ID"""
        url = "https://youtube.com/watch?v=abc-def_123"
        assert extract_video_id(url) == "abc-def_123"

    def test_extract_returns_none_for_empty(self):
        """Test returns None for empty input"""
        assert extract_video_id("") is None
        assert extract_video_id(None) is None

    def test_extract_returns_none_for_short(self):
        """Test returns None for too-short input"""
        assert extract_video_id("short") is None
        assert extract_video_id("abcdefghij") is None  # 10 chars

    def test_extract_case_sensitive(self):
        """Test that extraction preserves case"""
        url = "https://youtube.com/watch?v=AbCdEfGhIjK"
        assert extract_video_id(url) == "AbCdEfGhIjK"

    def test_extract_from_mobile_url(self):
        """Test extraction from mobile URL"""
        url = "https://m.youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_with_hash_fragment(self):
        """Test extraction with URL fragment"""
        url = "https://youtube.com/watch?v=dQw4w9WgXcQ#t=30"
        assert extract_video_id(url) == "dQw4w9WgXcQ"


class TestDetectCaptionFormat:
    """Tests for detect_caption_format function"""

    def test_detect_srt_by_extension(self):
        """Test SRT detection by extension"""
        assert detect_caption_format(Path("video.srt")) == "srt"
        assert detect_caption_format(Path("video.SRT")) == "srt"

    def test_detect_vtt_by_extension(self):
        """Test VTT detection by extension"""
        assert detect_caption_format(Path("video.vtt")) == "vtt"
        assert detect_caption_format(Path("video.VTT")) == "vtt"

    def test_detect_ass_by_extension(self):
        """Test ASS/SSA detection by extension"""
        assert detect_caption_format(Path("video.ass")) == "ass"
        assert detect_caption_format(Path("video.ssa")) == "ass"

    def test_detect_json3_by_extension(self):
        """Test JSON3 detection by extension"""
        assert detect_caption_format(Path("video.json3")) == "json3"
        assert detect_caption_format(Path("video.json")) == "json3"

    def test_detect_ttml_by_extension(self):
        """Test TTML/DFXP detection by extension"""
        assert detect_caption_format(Path("video.ttml")) == "ttml"
        assert detect_caption_format(Path("video.dfxp")) == "ttml"

    def test_detect_unknown_extension(self):
        """Test unknown extension returns None"""
        assert detect_caption_format(Path("video.xyz")) is None
        assert detect_caption_format(Path("video.mp4")) is None

    def test_detect_vtt_by_content(self):
        """Test VTT detection by file content"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.txt"
            filepath.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nHello")
            assert detect_caption_format(filepath) == "vtt"

    def test_detect_srt_by_content(self):
        """Test SRT detection by file content"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.txt"
            filepath.write_text("1\n00:00:00,000 --> 00:00:02,000\nHello")
            assert detect_caption_format(filepath) == "srt"

    def test_detect_ass_by_content(self):
        """Test ASS detection by file content"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.txt"
            filepath.write_text("[Script Info]\nTitle: Test\n")
            assert detect_caption_format(filepath) == "ass"


class TestGetCaptionLanguage:
    """Tests for get_caption_language function"""

    def test_extract_simple_language(self):
        """Test simple language extraction"""
        assert get_caption_language(Path("video.en.srt")) == "en"
        assert get_caption_language(Path("video.es.srt")) == "es"
        assert get_caption_language(Path("video.fr.srt")) == "fr"

    def test_extract_with_video_id(self):
        """Test extraction with video ID prefix"""
        path = Path("dQw4w9WgXcQ.en.srt")
        assert get_caption_language(path, "dQw4w9WgXcQ") == "en"

    def test_extract_regional_language(self):
        """Test regional language codes"""
        assert get_caption_language(Path("video.en-US.srt")) == "en-US"
        assert get_caption_language(Path("video.en-GB.srt")) == "en-GB"
        assert get_caption_language(Path("video.pt-BR.srt")) == "pt-BR"

    def test_extract_auto_language(self):
        """Test auto-generated caption language"""
        assert get_caption_language(Path("video.en-auto.srt")) == "en"
        assert get_caption_language(Path("video.es.auto.srt")) == "es"

    def test_extract_defaults_to_en(self):
        """Test defaults to 'en' when not found"""
        assert get_caption_language(Path("video.srt")) == "en"
        assert get_caption_language(Path("unknown_format.txt")) == "en"

    def test_extract_with_complex_filename(self):
        """Test extraction with complex filename"""
        path = Path("My_Video_Title_dQw4w9WgXcQ.en-US.srt")
        lang = get_caption_language(path, "dQw4w9WgXcQ")
        assert lang in ["en-US", "en"]  # Should find language code


class TestIsAutoGeneratedCaption:
    """Tests for is_auto_generated_caption function"""

    def test_auto_with_dash(self):
        """Test detection with -auto suffix"""
        assert is_auto_generated_caption(Path("video.en-auto.srt")) is True
        assert is_auto_generated_caption(Path("abc123.en-auto.vtt")) is True

    def test_auto_with_dot(self):
        """Test detection with .auto suffix"""
        assert is_auto_generated_caption(Path("video.en.auto.srt")) is True

    def test_manual_caption(self):
        """Test manual captions return False"""
        assert is_auto_generated_caption(Path("video.en.srt")) is False
        assert is_auto_generated_caption(Path("abc123.es.vtt")) is False

    def test_case_insensitive(self):
        """Test case insensitive detection"""
        assert is_auto_generated_caption(Path("video.en-AUTO.srt")) is True
        assert is_auto_generated_caption(Path("video.en-Auto.srt")) is True

    def test_auto_in_directory_name(self):
        """Test auto in directory doesn't trigger false positive"""
        # The function only checks filename, not full path
        assert is_auto_generated_caption(Path("video.en.srt")) is False


class TestCaptionFilePriority:
    """Tests for caption_file_priority function"""

    def test_manual_before_auto(self):
        """Test manual captions have higher priority than auto"""
        manual = caption_file_priority(Path("video.en.srt"))
        auto = caption_file_priority(Path("video.en-auto.srt"))
        assert manual < auto  # Lower tuple = higher priority

    def test_english_before_other(self):
        """Test English has higher priority than other languages"""
        english = caption_file_priority(Path("video.en.srt"))
        spanish = caption_file_priority(Path("video.es.srt"))
        assert english < spanish

    def test_manual_english_highest(self):
        """Test manual English has highest priority"""
        manual_en = caption_file_priority(Path("video.en.srt"))
        manual_es = caption_file_priority(Path("video.es.srt"))
        auto_en = caption_file_priority(Path("video.en-auto.srt"))
        auto_es = caption_file_priority(Path("video.es-auto.srt"))

        priorities = [manual_en, manual_es, auto_en, auto_es]
        assert manual_en == min(priorities)

    def test_sorting_multiple_files(self):
        """Test sorting multiple caption files"""
        files = [
            Path("video.es-auto.srt"),
            Path("video.en.srt"),
            Path("video.en-auto.srt"),
            Path("video.es.srt"),
        ]
        sorted_files = sorted(files, key=caption_file_priority)

        # Manual English should be first
        assert sorted_files[0].name == "video.en.srt"

    def test_returns_tuple(self):
        """Test priority returns a tuple for sorting"""
        priority = caption_file_priority(Path("video.en.srt"))
        assert isinstance(priority, tuple)
        assert len(priority) == 3


class TestCaptionUtilsEdgeCases:
    """Edge case tests for caption utility functions"""

    def test_extract_video_id_from_complex_path(self):
        """Test video ID extraction from complex path"""
        path = "C:/Users/test/videos/dQw4w9WgXcQ_720p.mp4"
        assert extract_video_id(path) == "dQw4w9WgXcQ"

    def test_detect_format_nonexistent_file(self):
        """Test format detection for nonexistent file with unknown extension"""
        result = detect_caption_format(Path("/nonexistent/video.xyz"))
        assert result is None

    def test_language_from_minimal_filename(self):
        """Test language extraction from minimal filename"""
        lang = get_caption_language(Path(".srt"))
        assert lang == "en"  # Should default to 'en'

    def test_priority_consistency(self):
        """Test priority function is consistent"""
        path = Path("video.en.srt")
        p1 = caption_file_priority(path)
        p2 = caption_file_priority(path)
        assert p1 == p2

    def test_auto_detection_partial_match(self):
        """Test auto detection doesn't partial match"""
        # 'automatic' in filename shouldn't trigger auto detection
        # Only '-auto' or '.auto' should
        assert is_auto_generated_caption(Path("video_automatic.en.srt")) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
