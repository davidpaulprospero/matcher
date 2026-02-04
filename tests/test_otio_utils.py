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
    parse_timecode_to_frames,
    get_confidence_color,
    create_clip_with_timewarp,
    _validate_clip_metadata,
    validate_timeline_clips,
    ClipValidationError,
    _has_problematic_path,
    _is_audio_only,
    AUDIO_ONLY_EXTS,
    NON_MEDIA_EXTS,
)
import numpy as np


class TestWindowsPathConversion:
    """Test Windows path conversion (now uses forward slashes for DaVinci)."""

    @pytest.mark.fast
    def test_to_windows_path_forward_slashes(self):
        """Test that forward slashes are preserved (DaVinci prefers them)."""
        path = "C:/Users/test/video.mp4"
        result = _to_windows_path(path)
        # Now converts TO forward slashes for DaVinci compatibility
        assert "/" in result or "\\" not in result.replace("\\\\", "")

    @pytest.mark.fast
    def test_to_windows_path_already_backslashes(self):
        """Test converting backslashes to forward slashes."""
        path = "C:\\Users\\test\\video.mp4"
        result = _to_windows_path(path)
        # Should convert to forward slashes
        assert "\\" not in result or result.count("/") > 0

    @pytest.mark.fast
    def test_to_windows_path_returns_absolute(self):
        """Test that result is an absolute path with forward slashes."""
        path = "C:/Users/test/video.mp4"
        result = _to_windows_path(path)
        # Should be absolute and use forward slashes
        assert ":" in result  # Has drive letter
        assert "/" in result  # Uses forward slashes


class TestPathFormatting:
    """Test path formatting for URLs."""

    @pytest.mark.fast
    def test_format_path_url_basic(self):
        """Test basic path URL formatting."""
        path = "C:/Videos/test.mp4"
        formatted = format_path_url(path)
        assert formatted is not None
        assert isinstance(formatted, str)
        # Returns plain path with forward slashes, not file:// URL
        assert "/" in formatted

    @pytest.mark.fast
    def test_format_path_url_windows_backslashes(self):
        """Test Windows path with backslashes."""
        path = "C:\\Users\\test\\video.mp4"
        formatted = format_path_url(path)
        assert formatted is not None
        # Should convert to forward slashes for URL
        assert "\\" not in formatted or path.startswith("\\\\?\\")

    @pytest.mark.fast
    def test_format_path_url_with_spaces(self):
        """Test path with spaces."""
        path = "C:/Videos/test video.mp4"
        formatted = format_path_url(path)
        assert formatted is not None
        # Returns plain path - spaces NOT encoded (DaVinci prefers unencoded)
        assert isinstance(formatted, str)

    @pytest.mark.fast
    def test_sanitize_path_for_url(self):
        """Test sanitizing path for URL."""
        path = "C:/Videos/test video.mp4"
        sanitized = sanitize_path_for_url(path)
        assert sanitized is not None
        # Should handle special characters
        assert isinstance(sanitized, str)

    @pytest.mark.fast
    def test_sanitize_path_for_url_double_slashes(self):
        """Test removing double slashes."""
        path = "C://Users//test//video.mp4"
        sanitized = sanitize_path_for_url(path)
        # Should not have consecutive slashes (except file://)
        assert "///" not in sanitized

    @pytest.mark.fast
    def test_encode_path_for_xml_url(self):
        """Test encoding path for XML URL."""
        path = "C:/Videos/test & more.mp4"
        encoded = encode_path_for_xml_url(path)
        assert encoded is not None
        # Function returns plain path - doesn't XML-encode (done elsewhere)
        assert isinstance(encoded, str)


class TestXMLEscaping:
    """Test XML escaping."""

    @pytest.mark.fast
    def test_escape_xml_ampersand(self):
        """Test escaping ampersand."""
        text = "Test & more"
        escaped = escape_xml(text)
        assert "&amp;" in escaped
        assert "&" not in escaped.replace("&amp;", "")

    @pytest.mark.fast
    def test_escape_xml_less_than(self):
        """Test escaping less than."""
        text = "x < 5"
        escaped = escape_xml(text)
        assert "&lt;" in escaped

    @pytest.mark.fast
    def test_escape_xml_greater_than(self):
        """Test escaping greater than."""
        text = "x > 5"
        escaped = escape_xml(text)
        assert "&gt;" in escaped

    @pytest.mark.fast
    def test_escape_xml_quotes(self):
        """Test escaping double quotes."""
        text = 'Say "hello"'
        escaped = escape_xml(text)
        assert "&quot;" in escaped

    @pytest.mark.fast
    def test_escape_xml_apostrophe(self):
        """Test escaping apostrophes."""
        text = "It's working"
        escaped = escape_xml(text)
        assert "&apos;" in escaped or "'" not in escaped

    @pytest.mark.fast
    def test_escape_xml_no_special_chars(self):
        """Test text without special characters."""
        text = "Plain text"
        escaped = escape_xml(text)
        assert escaped == text

    @pytest.mark.fast
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

    @pytest.mark.fast
    def test_to_python_type_numpy_int(self):
        """Test converting numpy int to Python int."""
        value = np.int64(42)
        result = _to_python_type(value)
        assert isinstance(result, int)
        assert result == 42

    @pytest.mark.fast
    def test_to_python_type_numpy_float(self):
        """Test converting numpy float to Python float."""
        value = np.float64(3.14)
        result = _to_python_type(value)
        assert isinstance(result, float)
        assert abs(result - 3.14) < 0.01

    @pytest.mark.fast
    def test_to_python_type_regular_int(self):
        """Test regular int passes through."""
        value = 42
        result = _to_python_type(value)
        assert result == 42

    @pytest.mark.fast
    def test_to_python_type_string(self):
        """Test string passes through."""
        value = "test"
        result = _to_python_type(value)
        assert result == "test"


class TestMetadataSanitization:
    """Test metadata sanitization for JSON serialization."""

    @pytest.mark.fast
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

    @pytest.mark.fast
    def test_sanitize_metadata_nested_dict(self):
        """Test sanitizing nested metadata."""
        metadata = {
            "outer": {
                "inner": np.float64(0.5)
            }
        }
        sanitized = _sanitize_metadata(metadata)
        assert isinstance(sanitized["outer"]["inner"], float)

    @pytest.mark.fast
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
    @pytest.mark.fast
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
    @pytest.mark.fast
    def test_get_media_duration_failure(self, mock_run):
        """Test duration extraction failure."""
        mock_run.return_value = Mock(returncode=1, stdout="", stderr="Error")

        duration = _get_media_duration("/path/to/video.mp4")
        assert duration is None

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_get_media_duration_invalid_output(self, mock_run):
        """Test handling invalid ffprobe output."""
        mock_run.return_value = Mock(returncode=0, stdout="invalid\n", stderr="")

        duration = _get_media_duration("/path/to/video.mp4")
        assert duration is None


class TestSegmentFileHandling:
    """Test segment file detection and offset extraction."""

    @pytest.mark.fast
    def test_is_segment_file_true(self):
        """Test detecting segment file with _0000.mp4 pattern."""
        result = is_segment_file("/path/to/video_0000.mp4")
        assert result is True

    @pytest.mark.fast
    def test_is_segment_file_false(self):
        """Test non-segment file."""
        result = is_segment_file("/path/to/video.mp4")
        assert result is False

    @pytest.mark.fast
    def test_is_segment_file_different_pattern(self):
        """Test segment file with different number pattern."""
        result = is_segment_file("/path/to/video_segment_0005.mp4")
        assert isinstance(result, bool)

    @pytest.mark.fast
    def test_get_segment_file_offset_basic(self):
        """Test extracting offset from segment filename with 4-digit pattern."""
        offset = get_segment_file_offset("/path/to/abc12345678_0045.mp4")
        assert offset == 45.0

    @pytest.mark.fast
    def test_get_segment_file_offset_no_offset(self):
        """Test regular (non-segment) file."""
        offset = get_segment_file_offset("/path/to/video.mp4")
        assert offset == 0.0

    @pytest.mark.fast
    def test_get_segment_file_offset_zero(self):
        """Test segment file with zero offset."""
        offset = get_segment_file_offset("/path/to/videoidhere_0000.mp4")
        assert offset == 0.0

    @pytest.mark.fast
    def test_get_segment_file_offset_large_value(self):
        """Test offset with large value (e.g., 2 hours)."""
        offset = get_segment_file_offset("/path/to/myvideofile_7200.mp4")
        assert offset == 7200.0


class TestTimecodeConversion:
    """Test timecode conversion."""

    @pytest.mark.fast
    def test_frames_to_tc_basic(self):
        """Test converting frames to timecode."""
        tc = frames_to_tc(300, fps=30.0)
        assert tc is not None
        assert isinstance(tc, str)
        # 300 frames at 30fps = 10 seconds = 00:00:10:00
        assert "10" in tc

    @pytest.mark.fast
    def test_frames_to_tc_zero(self):
        """Test zero frames."""
        tc = frames_to_tc(0, fps=30.0)
        assert "00:00:00:00" in tc

    @pytest.mark.fast
    def test_frames_to_tc_different_fps(self):
        """Test with different frame rates."""
        tc1 = frames_to_tc(60, fps=60.0)
        tc2 = frames_to_tc(30, fps=30.0)
        # Both should be 1 second
        assert tc1 is not None
        assert tc2 is not None

    @pytest.mark.fast
    def test_frames_to_tc_zero_frames(self):
        """Test 0 frames returns all-zero timecode."""
        assert frames_to_tc(0, fps=30.0) == "00:00:00:00"
        assert frames_to_tc(0, fps=24.0) == "00:00:00:00"

    @pytest.mark.fast
    def test_frames_to_tc_with_start_offset(self):
        """Test frames_to_tc with start_frame_offset (e.g. 01:00:00:00)."""
        # 01:00:00:00 at 30fps = 108000 frames offset
        offset = parse_timecode_to_frames("01:00:00:00", 30.0)
        assert offset == 108000
        tc = frames_to_tc(0, fps=30.0, start_frame_offset=offset)
        assert tc == "01:00:00:00"
        # 90 frames = 3 seconds at 30fps
        tc = frames_to_tc(90, fps=30.0, start_frame_offset=offset)
        assert tc == "01:00:03:00"

    @pytest.mark.fast
    def test_frames_to_tc_drop_frame_separator(self):
        """Test drop-frame separator (semicolon before frames)."""
        tc = frames_to_tc(15, fps=30.0, separator=';')
        assert tc == "00:00:00;15"
        # With offset
        offset = parse_timecode_to_frames("01:00:00;00", 30.0)
        tc = frames_to_tc(0, fps=30.0, start_frame_offset=offset, separator=';')
        assert tc == "01:00:00;00"

    @pytest.mark.fast
    def test_frames_to_tc_29_97_rounded(self):
        """Test with 29.97fps (rounded to int 29 internally)."""
        # 29.97 -> int(29.97) = 29 fps for frame counting
        tc = frames_to_tc(29, fps=29.97)
        # 29 frames at int(29.97)=29 fps = 1 second, 0 frames
        assert tc == "00:00:01:00"

    @pytest.mark.fast
    def test_parse_timecode_to_frames_standard(self):
        """Test parsing standard non-drop-frame timecode."""
        # 01:00:00:00 at 30fps = (1*3600 + 0*60 + 0) * 30 + 0 = 108000
        assert parse_timecode_to_frames("01:00:00:00", 30.0) == 108000
        # 00:01:00:00 at 30fps = 60 * 30 = 1800
        assert parse_timecode_to_frames("00:01:00:00", 30.0) == 1800

    @pytest.mark.fast
    def test_parse_timecode_to_frames_drop_frame(self):
        """Test parsing drop-frame timecode with semicolon separator."""
        # Should handle ';' same as ':'
        assert parse_timecode_to_frames("01:00:00;00", 30.0) == 108000
        assert parse_timecode_to_frames("00:00:01;15", 30.0) == 45

    @pytest.mark.fast
    def test_parse_timecode_to_frames_with_frame_remainder(self):
        """Test parsing timecode with non-zero frame component."""
        # 00:00:01:15 at 30fps = 1*30 + 15 = 45
        assert parse_timecode_to_frames("00:00:01:15", 30.0) == 45


class TestConfidenceColor:
    """Test confidence color mapping."""

    @pytest.mark.fast
    def test_confidence_color_high(self):
        """Test color for high confidence."""
        color = get_confidence_color(0.9)
        assert color is not None
        # High confidence should be green
        assert color == "GREEN"

    @pytest.mark.fast
    def test_confidence_color_medium(self):
        """Test color for medium confidence."""
        color = get_confidence_color(0.6)
        assert color == "CYAN"

    @pytest.mark.fast
    def test_confidence_color_low(self):
        """Test color for low confidence."""
        color = get_confidence_color(0.3)
        assert color == "ORANGE"

    @pytest.mark.fast
    def test_confidence_color_very_low(self):
        """Test color for very low confidence."""
        color = get_confidence_color(0.1)
        assert color == "RED"

    @pytest.mark.fast
    def test_confidence_color_zero(self):
        """Test color for zero confidence."""
        color = get_confidence_color(0.0)
        assert color == "RED"

    @pytest.mark.fast
    def test_confidence_color_boundary_cases(self):
        """Test boundary values for color mapping."""
        assert get_confidence_color(0.8) == "GREEN"
        assert get_confidence_color(0.6) == "CYAN"
        assert get_confidence_color(0.4) == "YELLOW"
        assert get_confidence_color(0.2) == "ORANGE"


class TestClipCreation:
    """Test OTIO clip creation with timewarp."""

    @pytest.mark.fast
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

    @pytest.mark.fast
    def test_create_clip_with_slowdown(self):
        """Test clip creation with source longer than target (trim approach).

        The function uses TRIM approach: source_range.duration = target_duration.
        No LinearTimeWarp is applied - the clip plays at normal speed for target_duration.
        """
        clip = create_clip_with_timewarp(
            name="SlowClip",
            source_path="C:/Videos/test.mp4",
            source_start=0.0,
            source_duration=10.0,
            target_duration=15.0,
            frame_rate=30.0
        )

        assert clip is not None
        # No timewarp effect - we use trim approach, not speed approach
        assert len(clip.effects) == 0
        # source_range.duration should equal target_duration
        assert clip.source_range.duration.value == 15.0 * 30.0  # 15s at 30fps

    @pytest.mark.fast
    def test_create_clip_with_speedup(self):
        """Test clip creation with source shorter than target (trim approach).

        The function uses TRIM approach: source_range.duration = target_duration.
        No LinearTimeWarp is applied - the clip plays at normal speed for target_duration.
        """
        clip = create_clip_with_timewarp(
            name="FastClip",
            source_path="C:/Videos/test.mp4",
            source_start=0.0,
            source_duration=10.0,
            target_duration=5.0,
            frame_rate=30.0
        )

        assert clip is not None
        # No timewarp effect - we use trim approach, not speed approach
        assert len(clip.effects) == 0
        # source_range.duration should equal target_duration
        assert clip.source_range.duration.value == 5.0 * 30.0  # 5s at 30fps

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
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


class TestOTIOTimingModel:
    """Test OTIO timing model compliance.

    Per OTIO docs: source_range MUST fit within available_range.
    These tests ensure our clip creation functions respect this constraint.
    """

    @pytest.mark.fast
    def test_source_range_within_available_range(self):
        """Test that source_range always fits within available_range."""
        clip = create_clip_with_timewarp(
            name="TestClip",
            source_path="C:/Videos/test.mp4",
            source_start=10.0,
            source_duration=5.0,
            target_duration=5.0,
            frame_rate=30.0,
            media_duration=60.0  # 60s video
        )

        # Get ranges
        avail_range = clip.media_reference.available_range
        src_range = clip.source_range

        # OTIO constraint: source_start + source_duration <= available_duration
        src_end = src_range.start_time.value + src_range.duration.value
        avail_end = avail_range.start_time.value + avail_range.duration.value

        assert src_end <= avail_end, (
            f"OTIO violation: source_range ({src_range.start_time.value}-{src_end}) "
            f"exceeds available_range ({avail_range.start_time.value}-{avail_end})"
        )

    @pytest.mark.fast
    def test_no_effects_means_normal_speed(self):
        """Test that clips without effects play at normal speed (100%)."""
        clip = create_clip_with_timewarp(
            name="NormalSpeedClip",
            source_path="C:/Videos/test.mp4",
            source_start=0.0,
            source_duration=5.0,
            target_duration=5.0,
            frame_rate=30.0
        )

        # No effects = normal playback speed
        assert len(clip.effects) == 0

        # Metadata should indicate 100% speed
        assert clip.metadata.get('time_scalar') == 1.0
        assert clip.metadata.get('speed_percent') == 100.0

    @pytest.mark.fast
    def test_metadata_contains_timing_info(self):
        """Test that clips have proper timing metadata for debugging."""
        clip = create_clip_with_timewarp(
            name="MetadataClip",
            source_path="C:/Videos/test.mp4",
            source_start=0.0,
            source_duration=10.0,
            target_duration=5.0,
            frame_rate=30.0
        )

        # Should have timing metadata
        assert 'target_duration' in clip.metadata
        assert 'source_duration' in clip.metadata
        assert 'time_scalar' in clip.metadata

        # Values should be correct
        assert clip.metadata['target_duration'] == 5.0
        assert clip.metadata['source_duration'] == 10.0
        assert clip.metadata['time_scalar'] == 1.0  # Always 1.0 (trim approach)


class TestClipMetadataValidation:
    """Test OTIO clip metadata validation functions."""

    @pytest.mark.fast
    def test_validate_clip_valid_clip(self):
        """Test validation passes for a well-formed clip."""
        clip = create_clip_with_timewarp(
            name="ValidClip",
            source_path="C:/Videos/test.mp4",
            source_start=0.0,
            source_duration=5.0,
            target_duration=5.0,
            frame_rate=30.0
        )
        errors = _validate_clip_metadata(clip)
        assert len(errors) == 0

    @pytest.mark.fast
    def test_validate_clip_negative_start_time(self):
        """Test validation catches negative start_time."""
        # Create a clip and manually set negative start_time
        clip = otio.schema.Clip(
            name="NegativeStartClip",
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(-30, 30.0),
                duration=otio.opentime.RationalTime(150, 30.0)
            ),
            media_reference=otio.schema.ExternalReference(
                target_url="C:/Videos/test.mp4"
            )
        )
        errors = _validate_clip_metadata(clip)
        assert len(errors) == 1
        assert errors[0].error_type == "negative_start_time"
        assert "-30" in errors[0].message

    @pytest.mark.fast
    def test_validate_clip_zero_duration(self):
        """Test validation catches zero duration."""
        clip = otio.schema.Clip(
            name="ZeroDurationClip",
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, 30.0),
                duration=otio.opentime.RationalTime(0, 30.0)
            ),
            media_reference=otio.schema.ExternalReference(
                target_url="C:/Videos/test.mp4"
            )
        )
        errors = _validate_clip_metadata(clip)
        assert len(errors) == 1
        assert errors[0].error_type == "non_positive_duration"

    @pytest.mark.fast
    def test_validate_clip_negative_duration(self):
        """Test validation catches negative duration."""
        clip = otio.schema.Clip(
            name="NegativeDurationClip",
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, 30.0),
                duration=otio.opentime.RationalTime(-10, 30.0)
            ),
            media_reference=otio.schema.ExternalReference(
                target_url="C:/Videos/test.mp4"
            )
        )
        errors = _validate_clip_metadata(clip)
        assert len(errors) == 1
        assert errors[0].error_type == "non_positive_duration"

    @pytest.mark.fast
    def test_validate_clip_empty_target_url(self):
        """Test validation catches empty target_url."""
        clip = otio.schema.Clip(
            name="EmptyUrlClip",
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, 30.0),
                duration=otio.opentime.RationalTime(150, 30.0)
            ),
            media_reference=otio.schema.ExternalReference(
                target_url=""
            )
        )
        errors = _validate_clip_metadata(clip)
        assert len(errors) == 1
        assert errors[0].error_type == "empty_target_url"

    @pytest.mark.fast
    def test_validate_clip_whitespace_target_url(self):
        """Test validation catches whitespace-only target_url."""
        clip = otio.schema.Clip(
            name="WhitespaceUrlClip",
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, 30.0),
                duration=otio.opentime.RationalTime(150, 30.0)
            ),
            media_reference=otio.schema.ExternalReference(
                target_url="   "
            )
        )
        errors = _validate_clip_metadata(clip)
        assert len(errors) == 1
        assert errors[0].error_type == "empty_target_url"

    @pytest.mark.fast
    def test_validate_clip_missing_media_reference(self):
        """Test validation catches missing media_reference.

        Note: OTIO creates MissingReference by default when no media_reference is provided.
        """
        clip = otio.schema.Clip(
            name="NoMediaRefClip",
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, 30.0),
                duration=otio.opentime.RationalTime(150, 30.0)
            )
            # No media_reference - OTIO creates MissingReference by default
        )
        errors = _validate_clip_metadata(clip)
        assert len(errors) == 1
        assert errors[0].error_type == "missing_media_reference"
        assert "MissingReference" in errors[0].message

    @pytest.mark.fast
    def test_validate_clip_missing_source_range(self):
        """Test validation catches missing source_range."""
        clip = otio.schema.Clip(
            name="NoSourceRangeClip",
            media_reference=otio.schema.ExternalReference(
                target_url="C:/Videos/test.mp4"
            )
        )
        errors = _validate_clip_metadata(clip)
        assert len(errors) == 1
        assert errors[0].error_type == "missing_source_range"

    @pytest.mark.fast
    def test_validate_clip_multiple_errors(self):
        """Test validation reports multiple errors."""
        clip = otio.schema.Clip(
            name="MultiErrorClip",
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(-30, 30.0),
                duration=otio.opentime.RationalTime(0, 30.0)
            ),
            media_reference=otio.schema.ExternalReference(
                target_url=""
            )
        )
        errors = _validate_clip_metadata(clip)
        assert len(errors) == 3
        error_types = [e.error_type for e in errors]
        assert "negative_start_time" in error_types
        assert "non_positive_duration" in error_types
        assert "empty_target_url" in error_types

    @pytest.mark.fast
    def test_validate_clip_unnamed_clip(self):
        """Test validation handles unnamed clips."""
        clip = otio.schema.Clip(
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(-30, 30.0),
                duration=otio.opentime.RationalTime(150, 30.0)
            ),
            media_reference=otio.schema.ExternalReference(
                target_url="C:/Videos/test.mp4"
            )
        )
        errors = _validate_clip_metadata(clip)
        assert len(errors) == 1
        # Should use "<unnamed>" as fallback
        assert "<unnamed>" in errors[0].message

    @pytest.mark.fast
    def test_clip_validation_error_repr(self):
        """Test ClipValidationError string representation."""
        error = ClipValidationError(
            clip_name="TestClip",
            error_type="negative_start_time",
            message="Clip 'TestClip' has negative start_time: -30"
        )
        repr_str = repr(error)
        assert "TestClip" in repr_str
        assert "negative_start_time" in repr_str


class TestTimelineClipsValidation:
    """Test timeline-wide clip validation."""

    @pytest.mark.fast
    def test_validate_timeline_all_valid(self):
        """Test validation passes for timeline with valid clips."""
        timeline = otio.schema.Timeline(name="ValidTimeline")
        track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)

        clip1 = create_clip_with_timewarp(
            name="Clip1",
            source_path="C:/Videos/test1.mp4",
            source_start=0.0,
            source_duration=5.0,
            target_duration=5.0
        )
        clip2 = create_clip_with_timewarp(
            name="Clip2",
            source_path="C:/Videos/test2.mp4",
            source_start=0.0,
            source_duration=5.0,
            target_duration=5.0
        )
        track.append(clip1)
        track.append(clip2)
        timeline.tracks.append(track)

        errors = validate_timeline_clips(timeline)
        assert len(errors) == 0

    @pytest.mark.fast
    def test_validate_timeline_with_invalid_clips(self):
        """Test validation finds errors across multiple tracks."""
        timeline = otio.schema.Timeline(name="InvalidTimeline")

        # Track 1 with invalid clip
        track1 = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
        invalid_clip = otio.schema.Clip(
            name="InvalidClip",
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(-10, 30.0),
                duration=otio.opentime.RationalTime(150, 30.0)
            ),
            media_reference=otio.schema.ExternalReference(
                target_url="C:/Videos/test.mp4"
            )
        )
        track1.append(invalid_clip)
        timeline.tracks.append(track1)

        # Track 2 with valid clip
        track2 = otio.schema.Track(name="V2", kind=otio.schema.TrackKind.Video)
        valid_clip = create_clip_with_timewarp(
            name="ValidClip",
            source_path="C:/Videos/test2.mp4",
            source_start=0.0,
            source_duration=5.0,
            target_duration=5.0
        )
        track2.append(valid_clip)
        timeline.tracks.append(track2)

        errors = validate_timeline_clips(timeline)
        assert len(errors) == 1
        assert errors[0].clip_name == "InvalidClip"

    @pytest.mark.fast
    def test_validate_timeline_empty_timeline(self):
        """Test validation handles empty timeline."""
        timeline = otio.schema.Timeline(name="EmptyTimeline")
        errors = validate_timeline_clips(timeline)
        assert len(errors) == 0

    @pytest.mark.fast
    def test_validate_timeline_skips_gaps(self):
        """Test validation ignores Gap items (only validates Clips)."""
        timeline = otio.schema.Timeline(name="TimelineWithGaps")
        track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)

        # Add a gap
        gap = otio.schema.Gap(
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, 30.0),
                duration=otio.opentime.RationalTime(30, 30.0)
            )
        )
        track.append(gap)

        # Add a valid clip
        clip = create_clip_with_timewarp(
            name="ValidClip",
            source_path="C:/Videos/test.mp4",
            source_start=0.0,
            source_duration=5.0,
            target_duration=5.0
        )
        track.append(clip)
        timeline.tracks.append(track)

        errors = validate_timeline_clips(timeline)
        assert len(errors) == 0

    @pytest.mark.fast
    def test_validate_timeline_multiple_errors_in_track(self):
        """Test validation aggregates errors from multiple clips in one track."""
        timeline = otio.schema.Timeline(name="MultiErrorTimeline")
        track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)

        # Two invalid clips
        invalid1 = otio.schema.Clip(
            name="Invalid1",
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(-10, 30.0),
                duration=otio.opentime.RationalTime(150, 30.0)
            ),
            media_reference=otio.schema.ExternalReference(
                target_url="C:/Videos/test1.mp4"
            )
        )
        invalid2 = otio.schema.Clip(
            name="Invalid2",
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, 30.0),
                duration=otio.opentime.RationalTime(0, 30.0)
            ),
            media_reference=otio.schema.ExternalReference(
                target_url="C:/Videos/test2.mp4"
            )
        )
        track.append(invalid1)
        track.append(invalid2)
        timeline.tracks.append(track)

        errors = validate_timeline_clips(timeline)
        assert len(errors) == 2
        clip_names = [e.clip_name for e in errors]
        assert "Invalid1" in clip_names
        assert "Invalid2" in clip_names


class TestHasProblematicPath:
    """Test _has_problematic_path handles unicode, replacement chars, and ASCII-only paths."""

    @pytest.mark.fast
    def test_ascii_path_is_not_problematic(self):
        """Test that a standard ASCII path is not flagged."""
        assert _has_problematic_path("C:/Videos/test_video.mp4") is False

    @pytest.mark.fast
    def test_ascii_path_with_spaces(self):
        """Test that ASCII path with spaces is not flagged."""
        assert _has_problematic_path("C:/My Videos/test video.mp4") is False

    @pytest.mark.fast
    def test_replacement_char_ufffd(self):
        """Test that unicode replacement character U+FFFD is detected."""
        assert _has_problematic_path("C:/Videos/test\ufffdvideo.mp4") is True

    @pytest.mark.fast
    def test_non_ascii_accented_chars(self):
        """Test that accented characters are detected as problematic."""
        assert _has_problematic_path("C:/Videos/caf\u00e9_video.mp4") is True

    @pytest.mark.fast
    def test_non_ascii_cjk_chars(self):
        """Test that CJK characters are detected as problematic."""
        assert _has_problematic_path("C:/Videos/\u4e2d\u6587_video.mp4") is True

    @pytest.mark.fast
    def test_emoji_in_path(self):
        """Test that emoji characters are detected as problematic."""
        assert _has_problematic_path("C:/Videos/\U0001f600_video.mp4") is True

    @pytest.mark.fast
    def test_empty_path(self):
        """Test that an empty path is not flagged (no problematic chars)."""
        assert _has_problematic_path("") is False

    @pytest.mark.fast
    def test_windows_backslash_path(self):
        """Test that Windows backslash paths are not flagged."""
        assert _has_problematic_path("C:\\Users\\test\\video.mp4") is False

    @pytest.mark.fast
    def test_mixed_replacement_and_ascii(self):
        """Test path with replacement char mixed into otherwise ASCII path."""
        assert _has_problematic_path("E:/Edit Job/client/stock/\ufffd_invalid.mp4") is True


class TestIsAudioOnly:
    """Test _is_audio_only detects audio-only file extensions."""

    @pytest.mark.fast
    def test_mp3_is_audio(self):
        assert _is_audio_only("video.mp3") is True

    @pytest.mark.fast
    def test_wav_is_audio(self):
        assert _is_audio_only("audio.wav") is True

    @pytest.mark.fast
    def test_mp4_is_not_audio(self):
        assert _is_audio_only("video.mp4") is False

    @pytest.mark.fast
    def test_mkv_is_not_audio(self):
        assert _is_audio_only("video.mkv") is False

    @pytest.mark.fast
    def test_case_insensitive(self):
        assert _is_audio_only("audio.MP3") is True
        assert _is_audio_only("audio.Flac") is True


class TestSharedConstants:
    """Test that shared constants are accessible from utils."""

    @pytest.mark.fast
    def test_audio_only_exts_contains_mp3(self):
        assert '.mp3' in AUDIO_ONLY_EXTS

    @pytest.mark.fast
    def test_non_media_exts_contains_srt(self):
        assert '.srt' in NON_MEDIA_EXTS

    @pytest.mark.fast
    def test_non_media_exts_contains_json(self):
        assert '.json' in NON_MEDIA_EXTS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
