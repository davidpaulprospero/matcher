"""
Unit tests for src/otio/utils.py

Tests path handling, type conversion, media utilities, and OTIO clip creation.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import opentimelineio as otio
import pytest

from src.otio.utils import (
    NumpyEncoder,
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
    create_clip_with_timewarp,
)


# ============================================================
# NumpyEncoder Tests
# ============================================================

class TestNumpyEncoder:
    """Test NumpyEncoder JSON serialization."""

    def test_encode_numpy_int32(self):
        """Test encoding numpy int32."""
        data = {'value': np.int32(42)}
        result = json.dumps(data, cls=NumpyEncoder)
        assert result == '{"value": 42}'

    def test_encode_numpy_int64(self):
        """Test encoding numpy int64."""
        data = {'value': np.int64(9999999999)}
        result = json.dumps(data, cls=NumpyEncoder)
        assert result == '{"value": 9999999999}'

    def test_encode_numpy_float32(self):
        """Test encoding numpy float32."""
        data = {'value': np.float32(3.14)}
        result = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(result)
        assert abs(parsed['value'] - 3.14) < 0.01

    def test_encode_numpy_float64(self):
        """Test encoding numpy float64."""
        data = {'value': np.float64(2.718281828)}
        result = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(result)
        assert abs(parsed['value'] - 2.718281828) < 0.000001

    def test_encode_numpy_array(self):
        """Test encoding numpy array."""
        data = {'values': np.array([1, 2, 3, 4, 5])}
        result = json.dumps(data, cls=NumpyEncoder)
        assert result == '{"values": [1, 2, 3, 4, 5]}'

    def test_encode_mixed_numpy_types(self):
        """Test encoding mixed numpy types."""
        data = {
            'int': np.int32(10),
            'float': np.float32(3.14),
            'array': np.array([1, 2, 3])
        }
        result = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(result)
        assert parsed['int'] == 10
        assert abs(parsed['float'] - 3.14) < 0.01
        assert parsed['array'] == [1, 2, 3]


# ============================================================
# Path Utility Tests
# ============================================================

class TestPathUtilities:
    """Test path handling functions."""

    def test_to_windows_path_basic(self):
        """Test basic Windows path conversion."""
        # This will resolve to actual paths, so we test the format
        result = _to_windows_path("test.mp4")
        # Should have backslashes on Windows
        assert '\\' in result or '/' not in result  # Windows style

    def test_format_path_url_basic(self):
        """Test path URL formatting."""
        result = format_path_url("test.mp4")
        # Should have forward slashes
        assert '\\' not in result
        assert '/' in result

    def test_sanitize_path_for_url_extended_length_prefix(self):
        """Test sanitization of Windows extended-length path prefix."""
        input_path = r"\\?\C:\Users\test\video.mp4"
        result = sanitize_path_for_url(input_path)
        assert not result.startswith('\\\\?\\')
        assert not result.startswith('//?/')
        assert 'C:/' in result
        assert '\\' not in result

    def test_sanitize_path_for_url_device_prefix(self):
        """Test sanitization of Windows device path prefix."""
        input_path = r"\\.\C:\Users\test\video.mp4"
        result = sanitize_path_for_url(input_path)
        assert not result.startswith('\\\\.\\')
        assert not result.startswith('//.//')
        assert 'C:/' in result

    def test_sanitize_path_for_url_forward_slash_prefix(self):
        """Test sanitization of forward slash extended prefix."""
        input_path = "//?/C:/Users/test/video.mp4"
        result = sanitize_path_for_url(input_path)
        assert not result.startswith('//?/')
        assert 'C:/' in result

    def test_sanitize_path_for_url_backslash_conversion(self):
        """Test backslash to forward slash conversion."""
        input_path = r"C:\Users\test\video.mp4"
        result = sanitize_path_for_url(input_path)
        assert '\\' not in result
        assert 'C:/Users/test/video.mp4' in result

    def test_sanitize_path_for_url_double_slash_removal(self):
        """Test removal of double slashes."""
        input_path = "C://Users//test//video.mp4"
        result = sanitize_path_for_url(input_path)
        assert '//' not in result
        assert 'C:/Users/test/video.mp4' in result

    def test_encode_path_for_xml_url(self):
        """Test XML path URL encoding."""
        input_path = r"\\?\C:\Users\test\video.mp4"
        result = encode_path_for_xml_url(input_path)
        # Should sanitize and return clean path
        assert not result.startswith('\\\\?\\')
        assert '\\' not in result
        assert 'C:/Users/test/video.mp4' in result

    def test_escape_xml_basic(self):
        """Test XML character escaping."""
        text = "Test <tag> & 'quotes' \"double\""
        result = escape_xml(text)
        assert result == "Test &lt;tag&gt; &amp; &apos;quotes&apos; &quot;double&quot;"

    def test_escape_xml_all_chars(self):
        """Test escaping all special XML characters."""
        text = "& < > \" '"
        result = escape_xml(text)
        assert result == "&amp; &lt; &gt; &quot; &apos;"


# ============================================================
# Type Conversion Tests
# ============================================================

class TestTypeConversion:
    """Test type conversion utilities."""

    def test_to_python_type_numpy_int(self):
        """Test converting numpy int to Python int."""
        result = _to_python_type(np.int32(42))
        assert isinstance(result, int)
        assert result == 42

    def test_to_python_type_numpy_float(self):
        """Test converting numpy float to Python float."""
        result = _to_python_type(np.float32(3.14))
        assert isinstance(result, float)
        assert abs(result - 3.14) < 0.01

    def test_to_python_type_numpy_bool(self):
        """Test converting numpy bool to Python bool."""
        result = _to_python_type(np.bool_(True))
        assert isinstance(result, bool)
        assert result is True

    def test_to_python_type_numpy_array(self):
        """Test converting numpy array to list."""
        arr = np.array([1, 2, 3, 4, 5])
        result = _to_python_type(arr)
        assert isinstance(result, list)
        assert result == [1, 2, 3, 4, 5]

    def test_to_python_type_list_recursive(self):
        """Test recursive list conversion."""
        data = [np.int32(1), np.float32(2.5), np.array([3, 4])]
        result = _to_python_type(data)
        assert isinstance(result, list)
        assert isinstance(result[0], int)
        assert isinstance(result[1], float)
        assert isinstance(result[2], list)

    def test_to_python_type_dict_recursive(self):
        """Test recursive dict conversion."""
        data = {
            'int': np.int32(10),
            'float': np.float32(3.14),
            'nested': {'array': np.array([1, 2, 3])}
        }
        result = _to_python_type(data)
        assert isinstance(result['int'], int)
        assert isinstance(result['float'], float)
        assert isinstance(result['nested']['array'], list)

    def test_to_python_type_none(self):
        """Test handling None value."""
        result = _to_python_type(None)
        assert result is None

    def test_to_python_type_native_types(self):
        """Test native types pass through unchanged."""
        assert _to_python_type(42) == 42
        assert _to_python_type(3.14) == 3.14
        assert _to_python_type("text") == "text"
        assert _to_python_type(True) is True

    def test_sanitize_metadata(self):
        """Test metadata sanitization."""
        metadata = {
            'confidence': np.float32(0.85),
            'index': np.int32(5),
            'values': np.array([1, 2, 3]),
            'text': "normal string"
        }
        result = _sanitize_metadata(metadata)
        assert isinstance(result['confidence'], float)
        assert isinstance(result['index'], int)
        assert isinstance(result['values'], list)
        assert isinstance(result['text'], str)


# ============================================================
# Media Utility Tests
# ============================================================

class TestMediaUtilities:
    """Test media file utilities."""

    @patch('src.otio.utils.subprocess.run')
    def test_get_media_duration_success(self, mock_run):
        """Test successful media duration retrieval."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "125.5\n"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        duration = _get_media_duration("/path/to/video.mp4")

        assert duration == 125.5
        mock_run.assert_called_once()
        # Check that ffprobe was called
        args = mock_run.call_args[0][0]
        assert args[0] == 'ffprobe'

    @patch('src.otio.utils.subprocess.run')
    def test_get_media_duration_ffprobe_error(self, mock_run):
        """Test handling ffprobe error."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error message"
        mock_run.return_value = mock_result

        duration = _get_media_duration("/path/to/video.mp4")

        assert duration is None

    @patch('src.otio.utils.subprocess.run')
    def test_get_media_duration_ffprobe_not_found(self, mock_run):
        """Test handling ffprobe not found."""
        mock_run.side_effect = FileNotFoundError()

        duration = _get_media_duration("/path/to/video.mp4")

        assert duration is None

    @patch('src.otio.utils.subprocess.run')
    def test_get_media_duration_timeout(self, mock_run):
        """Test handling subprocess timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired('ffprobe', 10)

        duration = _get_media_duration("/path/to/video.mp4")

        assert duration is None

    def test_get_media_duration_no_path(self):
        """Test handling empty path."""
        duration = _get_media_duration(None)
        assert duration is None

    def test_get_segment_file_offset_standard_format(self):
        """Test extracting segment offset from filename."""
        # Standard format: video_id_NNNN.mp4
        assert get_segment_file_offset("abc12345678_0045.mp4") == 45.0
        assert get_segment_file_offset("xyz98765432_0120.mp4") == 120.0
        assert get_segment_file_offset("test_video_0000.mp4") == 0.0

    def test_get_segment_file_offset_various_patterns(self):
        """Test segment offset with various filename patterns."""
        assert get_segment_file_offset("longervideoname_0030.mp4") == 30.0
        assert get_segment_file_offset("/path/to/video_0015.mp4") == 15.0

    def test_get_segment_file_offset_non_segment_file(self):
        """Test non-segment files return 0."""
        assert get_segment_file_offset("regular_video.mp4") == 0.0
        assert get_segment_file_offset("no_pattern_here.mp4") == 0.0
        assert get_segment_file_offset("test_video.mp4") == 0.0

    def test_is_segment_file_with_offset(self):
        """Test segment file detection with offset."""
        assert is_segment_file("video_0045.mp4") is True
        assert is_segment_file("video_0120.mp4") is True

    def test_is_segment_file_zero_offset(self):
        """Test segment file detection with zero offset."""
        assert is_segment_file("video_0000.mp4") is True

    def test_is_segment_file_non_segment(self):
        """Test non-segment file detection."""
        assert is_segment_file("regular_video.mp4") is False
        assert is_segment_file("test.mp4") is False


# ============================================================
# Formatting Utility Tests
# ============================================================

class TestFormattingUtilities:
    """Test formatting utilities."""

    def test_frames_to_tc_basic(self):
        """Test basic frame to timecode conversion."""
        # 30 fps: 900 frames = 30 seconds = 00:00:30:00
        result = frames_to_tc(900, 30.0)
        assert result == "00:00:30:00"

    def test_frames_to_tc_with_hours(self):
        """Test timecode with hours."""
        # 30 fps: 108000 frames = 3600 seconds = 1 hour
        result = frames_to_tc(108000, 30.0)
        assert result == "01:00:00:00"

    def test_frames_to_tc_with_frame_remainder(self):
        """Test timecode with frame remainder."""
        # 30 fps: 925 frames = 30 seconds + 25 frames
        result = frames_to_tc(925, 30.0)
        assert result == "00:00:30:25"

    def test_frames_to_tc_zero_frames(self):
        """Test timecode with zero frames."""
        result = frames_to_tc(0, 30.0)
        assert result == "00:00:00:00"

    def test_frames_to_tc_different_framerate(self):
        """Test timecode with different frame rate."""
        # 24 fps: 240 frames = 10 seconds
        result = frames_to_tc(240, 24.0)
        assert result == "00:00:10:00"

    def test_get_confidence_color_high(self):
        """Test color for high confidence (>= 0.8)."""
        assert get_confidence_color(0.9) == "GREEN"
        assert get_confidence_color(0.8) == "GREEN"

    def test_get_confidence_color_medium_high(self):
        """Test color for medium-high confidence (>= 0.6)."""
        assert get_confidence_color(0.7) == "CYAN"
        assert get_confidence_color(0.6) == "CYAN"

    def test_get_confidence_color_medium(self):
        """Test color for medium confidence (>= 0.4)."""
        assert get_confidence_color(0.5) == "YELLOW"
        assert get_confidence_color(0.4) == "YELLOW"

    def test_get_confidence_color_medium_low(self):
        """Test color for medium-low confidence (>= 0.2)."""
        assert get_confidence_color(0.3) == "ORANGE"
        assert get_confidence_color(0.2) == "ORANGE"

    def test_get_confidence_color_low(self):
        """Test color for low confidence (< 0.2)."""
        assert get_confidence_color(0.1) == "RED"
        assert get_confidence_color(0.0) == "RED"


# ============================================================
# Clip Creation Tests
# ============================================================

class TestClipCreation:
    """Test OTIO clip creation with timewarp."""

    def test_create_clip_with_timewarp_basic(self):
        """Test basic clip creation."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            temp_path = f.name

        try:
            clip = create_clip_with_timewarp(
                name="Test Clip",
                source_path=temp_path,
                source_start=0.0,
                source_duration=10.0,
                target_duration=10.0,
                frame_rate=30.0
            )

            assert isinstance(clip, otio.schema.Clip)
            assert clip.name == "Test Clip"
            assert isinstance(clip.media_reference, otio.schema.ExternalReference)
            assert clip.source_range.duration.value == 300  # 10 seconds * 30 fps
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_create_clip_with_timewarp_speed_up(self):
        """Test clip with speed up (source > target)."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            temp_path = f.name

        try:
            clip = create_clip_with_timewarp(
                name="Fast Clip",
                source_path=temp_path,
                source_start=0.0,
                source_duration=10.0,  # 10 second source
                target_duration=5.0,    # Fit into 5 seconds
                frame_rate=30.0
            )

            # Should have LinearTimeWarp with time_scalar = 10/5 = 2.0 (speed up 2x)
            assert len(clip.effects) > 0
            time_warp = clip.effects[0]
            assert isinstance(time_warp, otio.schema.LinearTimeWarp)
            assert abs(time_warp.time_scalar - 2.0) < 0.01

            # Metadata should reflect speed
            assert 'time_scalar' in clip.metadata
            assert abs(clip.metadata['time_scalar'] - 2.0) < 0.01
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_create_clip_with_timewarp_slow_down(self):
        """Test clip with slow down (source < target)."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            temp_path = f.name

        try:
            clip = create_clip_with_timewarp(
                name="Slow Clip",
                source_path=temp_path,
                source_start=0.0,
                source_duration=5.0,   # 5 second source
                target_duration=10.0,  # Stretch to 10 seconds
                frame_rate=30.0
            )

            # Should have LinearTimeWarp with time_scalar = 5/10 = 0.5 (slow down 2x)
            assert len(clip.effects) > 0
            time_warp = clip.effects[0]
            assert isinstance(time_warp, otio.schema.LinearTimeWarp)
            assert abs(time_warp.time_scalar - 0.5) < 0.01
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_create_clip_with_timewarp_no_speed_change(self):
        """Test clip with no speed change (source == target)."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            temp_path = f.name

        try:
            clip = create_clip_with_timewarp(
                name="Normal Speed Clip",
                source_path=temp_path,
                source_start=0.0,
                source_duration=10.0,
                target_duration=10.0,
                frame_rate=30.0
            )

            # No time warp should be applied (within 1% threshold)
            assert len(clip.effects) == 0
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_create_clip_with_timewarp_metadata(self):
        """Test clip with custom metadata."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            temp_path = f.name

        try:
            metadata = {
                'confidence': 0.85,
                'segment_id': 'S001',
                'reasoning': 'Test reasoning'
            }
            clip = create_clip_with_timewarp(
                name="Metadata Clip",
                source_path=temp_path,
                source_start=0.0,
                source_duration=10.0,
                target_duration=10.0,
                frame_rate=30.0,
                metadata=metadata
            )

            assert clip.metadata['confidence'] == 0.85
            assert clip.metadata['segment_id'] == 'S001'
            assert clip.metadata['reasoning'] == 'Test reasoning'
            assert 'Resolve_OTIO' in clip.metadata
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_create_clip_with_timewarp_source_offset(self):
        """Test clip with non-zero source start."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            temp_path = f.name

        try:
            clip = create_clip_with_timewarp(
                name="Offset Clip",
                source_path=temp_path,
                source_start=15.0,  # Start at 15 seconds
                source_duration=10.0,
                target_duration=10.0,
                frame_rate=30.0
            )

            # Source range should start at 15 seconds = 450 frames
            assert clip.source_range.start_time.value == 450
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_create_clip_with_timewarp_unique_media_name(self):
        """Test that media reference has unique name including folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "subfolder" / "test_video.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.touch()

            clip = create_clip_with_timewarp(
                name="Unique Name Clip",
                source_path=str(video_path),
                source_start=0.0,
                source_duration=10.0,
                target_duration=10.0,
                frame_rate=30.0
            )

            # Media reference name should include folder to prevent conflicts
            assert 'subfolder' in clip.media_reference.name
            assert 'test_video.mp4' in clip.media_reference.name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
