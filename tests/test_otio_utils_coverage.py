"""
Additional tests for src/otio/utils.py to improve coverage.

Focuses on:
- NumpyEncoder edge cases (lines 24-30)
- sanitize_path_for_url prefix handling (lines 83-84, 88)
- _to_python_type edge cases (lines 141, 156-163)
- _get_media_duration error paths (lines 193, 215)
- create_clip_with_timewarp (line 370)
"""

import pytest
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

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


# ============================================================================
# Test NumpyEncoder (lines 21-30)
# ============================================================================

class TestNumpyEncoderExtended:
    """Extended tests for NumpyEncoder"""

    @pytest.mark.fast
    def test_encode_numpy_int32(self):
        """Test encoding numpy int32"""
        data = {'value': np.int32(42)}
        result = json.dumps(data, cls=NumpyEncoder)
        assert '"value": 42' in result

    @pytest.mark.fast
    def test_encode_numpy_int64(self):
        """Test encoding numpy int64"""
        data = {'value': np.int64(123456789)}
        result = json.dumps(data, cls=NumpyEncoder)
        assert '123456789' in result

    @pytest.mark.fast
    def test_encode_numpy_float32(self):
        """Test encoding numpy float32"""
        data = {'value': np.float32(3.14)}
        result = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(result)
        assert abs(parsed['value'] - 3.14) < 0.01

    @pytest.mark.fast
    def test_encode_numpy_float64(self):
        """Test encoding numpy float64"""
        data = {'value': np.float64(2.718281828)}
        result = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(result)
        assert abs(parsed['value'] - 2.718) < 0.01

    @pytest.mark.fast
    def test_encode_numpy_array(self):
        """Test encoding numpy array"""
        data = {'values': np.array([1, 2, 3])}
        result = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(result)
        assert parsed['values'] == [1, 2, 3]

    @pytest.mark.fast
    def test_encode_fallback_to_default(self):
        """Test fallback to default encoder for non-numpy types"""
        data = {'value': 'string'}
        result = json.dumps(data, cls=NumpyEncoder)
        assert '"value": "string"' in result

    @pytest.mark.fast
    def test_encode_unsupported_type_raises(self):
        """Test unsupported types raise TypeError"""
        class CustomClass:
            pass

        data = {'value': CustomClass()}

        with pytest.raises(TypeError):
            json.dumps(data, cls=NumpyEncoder)


# ============================================================================
# Test sanitize_path_for_url extended prefixes (lines 73-88)
# ============================================================================

class TestSanitizePathForUrlExtended:
    """Extended tests for sanitize_path_for_url"""

    @pytest.mark.fast
    def test_remove_standard_extended_length_prefix(self):
        """Test removing \\\\?\\ prefix"""
        path = "\\\\?\\C:\\Users\\test\\video.mp4"
        result = sanitize_path_for_url(path)
        assert result == "C:/Users/test/video.mp4"
        assert "?" not in result

    @pytest.mark.fast
    def test_remove_device_form_prefix(self):
        """Test removing \\\\.\\  device prefix"""
        path = "\\\\.\\C:\\Users\\test\\video.mp4"
        result = sanitize_path_for_url(path)
        assert result == "C:/Users/test/video.mp4"

    @pytest.mark.fast
    def test_remove_forward_slash_form(self):
        """Test removing //?/ prefix"""
        path = "//?/C:/Users/test/video.mp4"
        result = sanitize_path_for_url(path)
        assert result == "C:/Users/test/video.mp4"

    @pytest.mark.fast
    def test_remove_device_forward_slash(self):
        """Test removing //.// prefix"""
        path = "//./C:/Users/test/video.mp4"
        result = sanitize_path_for_url(path)
        # Remove double slashes
        assert "//" not in result or result.startswith("//")

    @pytest.mark.fast
    def test_remove_single_backslash_form(self):
        """Test removing \\?\\ prefix"""
        path = "\\?\\C:\\test.mp4"
        result = sanitize_path_for_url(path)
        assert "?" not in result

    @pytest.mark.fast
    def test_remove_question_backslash_start(self):
        """Test removing ?\\ at start after conversion"""
        path = "?\\C:\\test.mp4"
        result = sanitize_path_for_url(path)
        assert not result.startswith("?")

    @pytest.mark.fast
    def test_remove_question_slash_start(self):
        """Test removing ?/ at start"""
        path = "?/C:/test.mp4"
        result = sanitize_path_for_url(path)
        assert not result.startswith("?")

    @pytest.mark.fast
    def test_remove_double_slashes(self):
        """Test removing double slashes in path"""
        path = "C://Users//test//video.mp4"
        result = sanitize_path_for_url(path)
        assert "//" not in result

    @pytest.mark.fast
    def test_path_object_input(self):
        """Test Path object input"""
        path = Path("C:/Users/test/video.mp4")
        result = sanitize_path_for_url(path)
        assert isinstance(result, str)
        assert "video.mp4" in result


# ============================================================================
# Test _to_python_type edge cases (lines 138-173)
# ============================================================================

class TestToPythonTypeExtended:
    """Extended tests for _to_python_type"""

    @pytest.mark.fast
    def test_convert_none(self):
        """Test None passes through"""
        result = _to_python_type(None)
        assert result is None

    @pytest.mark.fast
    def test_convert_numpy_float(self):
        """Test numpy float conversion"""
        result = _to_python_type(np.float64(3.14))
        assert isinstance(result, float)
        assert abs(result - 3.14) < 0.001

    @pytest.mark.fast
    def test_convert_numpy_int(self):
        """Test numpy int conversion"""
        result = _to_python_type(np.int64(42))
        assert isinstance(result, int)
        assert result == 42

    @pytest.mark.fast
    def test_convert_numpy_bool(self):
        """Test numpy bool conversion"""
        result = _to_python_type(np.bool_(True))
        assert isinstance(result, bool)
        assert result is True

    @pytest.mark.fast
    def test_convert_numpy_str(self):
        """Test numpy string conversion"""
        result = _to_python_type(np.str_("test"))
        assert isinstance(result, str)
        assert result == "test"

    @pytest.mark.fast
    def test_convert_numpy_array_via_tolist(self):
        """Test numpy array conversion via tolist"""
        arr = np.array([1, 2, 3])
        result = _to_python_type(arr)
        assert isinstance(result, list)
        assert result == [1, 2, 3]

    @pytest.mark.fast
    def test_convert_nested_list(self):
        """Test nested list conversion"""
        data = [np.int64(1), np.float64(2.5), "string"]
        result = _to_python_type(data)
        assert result == [1, 2.5, "string"]
        assert isinstance(result[0], int)
        assert isinstance(result[1], float)

    @pytest.mark.fast
    def test_convert_nested_dict(self):
        """Test nested dict conversion"""
        data = {'a': np.int64(1), 'b': np.float64(2.5), 'c': 'string'}
        result = _to_python_type(data)
        assert result == {'a': 1, 'b': 2.5, 'c': 'string'}
        assert isinstance(result['a'], int)

    @pytest.mark.fast
    def test_convert_regular_python_type(self):
        """Test regular Python type passes through"""
        result = _to_python_type("regular string")
        assert result == "regular string"


# ============================================================================
# Test _get_media_duration error paths (lines 185-219)
# ============================================================================

class TestGetMediaDurationErrors:
    """Test _get_media_duration error handling"""

    @pytest.mark.fast
    def test_get_duration_empty_path(self):
        """Test with empty path"""
        result = _get_media_duration("")
        assert result is None

    @pytest.mark.fast
    def test_get_duration_none_path(self):
        """Test with None path"""
        result = _get_media_duration(None)
        assert result is None

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_get_duration_success(self, mock_run):
        """Test successful duration extraction"""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "123.45\n"
        mock_run.return_value = mock_result

        result = _get_media_duration("/test/video.mp4")

        assert result == 123.45

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_get_duration_ffprobe_fails(self, mock_run):
        """Test when ffprobe returns non-zero"""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error message"
        mock_run.return_value = mock_result

        result = _get_media_duration("/test/video.mp4")

        assert result is None

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_get_duration_empty_output(self, mock_run):
        """Test when ffprobe returns empty output"""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        result = _get_media_duration("/test/video.mp4")

        assert result is None

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_get_duration_ffprobe_not_found(self, mock_run):
        """Test when ffprobe is not installed"""
        mock_run.side_effect = FileNotFoundError("ffprobe not found")

        result = _get_media_duration("/test/video.mp4")

        assert result is None

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_get_duration_generic_exception(self, mock_run):
        """Test generic exception handling"""
        mock_run.side_effect = Exception("Unexpected error")

        result = _get_media_duration("/test/video.mp4")

        assert result is None


# ============================================================================
# Test get_segment_file_offset
# ============================================================================

class TestGetSegmentFileOffset:
    """Test get_segment_file_offset function"""

    @pytest.mark.fast
    def test_standard_segment_filename(self):
        """Test standard YouTube segment filename"""
        result = get_segment_file_offset("abc12345678_0045.mp4")
        assert result == 45.0

    @pytest.mark.fast
    def test_segment_at_start(self):
        """Test segment starting at 0"""
        result = get_segment_file_offset("video_id123_0000.mp4")
        assert result == 0.0

    @pytest.mark.fast
    def test_segment_high_offset(self):
        """Test segment with high offset"""
        result = get_segment_file_offset("abc12345678_0120.mp4")
        assert result == 120.0

    @pytest.mark.fast
    def test_non_segment_filename(self):
        """Test regular video filename"""
        result = get_segment_file_offset("regular_video.mp4")
        assert result == 0.0

    @pytest.mark.fast
    def test_longer_video_id(self):
        """Test with longer video ID format"""
        result = get_segment_file_offset("longer_video_id_here_0060.mp4")
        assert result == 60.0


# ============================================================================
# Test is_segment_file
# ============================================================================

class TestIsSegmentFile:
    """Test is_segment_file function"""

    @pytest.mark.fast
    def test_segment_file(self):
        """Test identifies segment file"""
        assert is_segment_file("abc12345678_0045.mp4") is True

    @pytest.mark.fast
    def test_segment_at_zero(self):
        """Test identifies segment at 0"""
        assert is_segment_file("abc12345678_0000.mp4") is True

    @pytest.mark.fast
    def test_regular_file(self):
        """Test regular file is not segment"""
        assert is_segment_file("regular_video.mp4") is False


# ============================================================================
# Test frames_to_tc
# ============================================================================

class TestFramesToTimecode:
    """Test frames_to_tc function"""

    @pytest.mark.fast
    def test_zero_frames(self):
        """Test zero frames"""
        result = frames_to_tc(0, fps=30.0)
        assert result == "00:00:00:00"

    @pytest.mark.fast
    def test_one_second(self):
        """Test one second worth of frames"""
        result = frames_to_tc(30, fps=30.0)
        assert result == "00:00:01:00"

    @pytest.mark.fast
    def test_one_minute(self):
        """Test one minute"""
        result = frames_to_tc(30 * 60, fps=30.0)
        assert result == "00:01:00:00"

    @pytest.mark.fast
    def test_one_hour(self):
        """Test one hour"""
        result = frames_to_tc(30 * 60 * 60, fps=30.0)
        assert result == "01:00:00:00"

    @pytest.mark.fast
    def test_partial_frames(self):
        """Test partial frame count"""
        result = frames_to_tc(95, fps=30.0)
        # 95 frames at 30fps = 3.166s = 00:00:03:05
        assert result == "00:00:03:05"


# ============================================================================
# Test get_confidence_color
# ============================================================================

class TestGetConfidenceColor:
    """Test get_confidence_color function"""

    @pytest.mark.fast
    def test_high_confidence(self):
        """Test high confidence (>= 0.8)"""
        assert get_confidence_color(0.9) == "GREEN"
        assert get_confidence_color(0.8) == "GREEN"

    @pytest.mark.fast
    def test_medium_high_confidence(self):
        """Test medium-high confidence (0.6-0.8)"""
        assert get_confidence_color(0.7) == "CYAN"
        assert get_confidence_color(0.6) == "CYAN"

    @pytest.mark.fast
    def test_medium_confidence(self):
        """Test medium confidence (0.4-0.6)"""
        assert get_confidence_color(0.5) == "YELLOW"
        assert get_confidence_color(0.4) == "YELLOW"

    @pytest.mark.fast
    def test_low_confidence(self):
        """Test low confidence (0.2-0.4)"""
        assert get_confidence_color(0.3) == "ORANGE"
        assert get_confidence_color(0.2) == "ORANGE"

    @pytest.mark.fast
    def test_very_low_confidence(self):
        """Test very low confidence (< 0.2)"""
        assert get_confidence_color(0.1) == "RED"
        assert get_confidence_color(0.0) == "RED"


# ============================================================================
# Test create_clip_with_timewarp (line 370)
# ============================================================================

class TestCreateClipWithTimewarp:
    """Test create_clip_with_timewarp function"""

    @pytest.mark.fast
    def test_create_clip_basic(self):
        """Test basic clip creation"""
        clip = create_clip_with_timewarp(
            name="Test Clip",
            source_path="E:/videos/test.mp4",
            source_start=0.0,
            source_duration=5.0,
            target_duration=5.0,
            frame_rate=30.0
        )

        assert clip.name == "Test Clip"
        assert clip.source_range is not None

    @pytest.mark.fast
    def test_create_clip_with_speed_up(self):
        """Test clip with speed increase (source > target) — no timewarp, trim approach"""
        clip = create_clip_with_timewarp(
            name="Speed Up",
            source_path="E:/videos/test.mp4",
            source_start=0.0,
            source_duration=10.0,
            target_duration=5.0,
            frame_rate=30.0
        )

        # No LinearTimeWarp — trim approach sets source_range.duration = target_duration
        assert len(clip.effects) == 0
        assert clip.source_range.duration.value == 150  # 5.0s * 30fps

    @pytest.mark.fast
    def test_create_clip_with_slow_down(self):
        """Test clip with speed decrease (source < target) — no timewarp when media is sufficient"""
        clip = create_clip_with_timewarp(
            name="Slow Down",
            source_path="E:/videos/test.mp4",
            source_start=0.0,
            source_duration=5.0,
            target_duration=10.0,
            frame_rate=30.0
        )

        # No LinearTimeWarp — media is estimated large enough, trim approach suffices
        assert len(clip.effects) == 0
        assert clip.source_range.duration.value == 300  # 10.0s * 30fps

    @pytest.mark.fast
    def test_create_clip_short_media_timewarp(self):
        """Test clip where media is shorter than target — keeps target duration for DaVinci sync"""
        clip = create_clip_with_timewarp(
            name="Short Media",
            source_path="E:/videos/test.mp4",
            source_start=0.0,
            source_duration=1.3,
            target_duration=16.0,
            frame_rate=30.0,
            media_duration=1.3  # Media genuinely shorter than target
        )

        # DaVinci may not interpret LinearTimeWarp, so source_range.duration
        # is kept at target_duration to ensure correct timeline positioning.
        # DaVinci will freeze the last frame for the overshoot.
        assert len(clip.effects) == 0  # No timewarp — rely on source_range.duration
        # source_range.duration = target (16s * 30fps = 480 frames)
        assert clip.source_range.duration.value == round(16.0 * 30)
        # Metadata still contains speed info for reference
        expected_scalar = 1.3 / 16.0
        assert abs(clip.metadata['speed_percent'] - expected_scalar * 100) < 1.0

    @pytest.mark.fast
    def test_create_clip_no_timewarp_needed(self):
        """Test clip with matching durations (no timewarp needed)"""
        clip = create_clip_with_timewarp(
            name="No Change",
            source_path="E:/videos/test.mp4",
            source_start=0.0,
            source_duration=5.0,
            target_duration=5.0,
            frame_rate=30.0
        )

        # No timewarp needed when durations match
        assert len(clip.effects) == 0

    @pytest.mark.fast
    def test_create_clip_very_short_duration(self):
        """Test clip with very short duration (minimum 1 frame)"""
        clip = create_clip_with_timewarp(
            name="Short",
            source_path="E:/videos/test.mp4",
            source_start=0.0,
            source_duration=0.01,
            target_duration=0.001,
            frame_rate=30.0
        )

        # Should have at least 1 frame duration
        duration_value = clip.source_range.duration.value
        assert duration_value >= 1

    @pytest.mark.fast
    def test_create_clip_with_metadata(self):
        """Test clip with custom metadata"""
        clip = create_clip_with_timewarp(
            name="With Metadata",
            source_path="E:/videos/test.mp4",
            source_start=0.0,
            source_duration=5.0,
            target_duration=5.0,
            frame_rate=30.0,
            metadata={'confidence': 0.9, 'strategy': 'primary'}
        )

        assert 'confidence' in clip.metadata
        assert clip.metadata['confidence'] == 0.9

    @pytest.mark.fast
    def test_create_clip_with_media_duration(self):
        """Test clip with explicit media duration"""
        clip = create_clip_with_timewarp(
            name="Known Duration",
            source_path="E:/videos/test.mp4",
            source_start=10.0,
            source_duration=5.0,
            target_duration=5.0,
            frame_rate=30.0,
            media_duration=120.0  # 2 minute video
        )

        # available_range should reflect media_duration
        available = clip.media_reference.available_range
        assert available.duration.value == 120.0 * 30.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
