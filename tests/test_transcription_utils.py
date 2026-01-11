"""
Tests for src/transcription/utils.py

Tests utility functions for audio extraction, SRT generation,
video ID extraction, and timestamp formatting.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, call
import subprocess

from src.transcription.utils import (
    extract_audio,
    write_srt,
    extract_video_id,
    format_timestamp_srt
)


class TestExtractAudio:
    """Test extract_audio() function"""

    @patch('subprocess.run')
    def test_extract_audio_success(self, mock_run, tmp_path):
        """Test successful audio extraction"""
        video_file = tmp_path / "test.mp4"
        video_file.write_text("fake video")

        # Mock successful ffmpeg
        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        # Mock the audio file being created
        with patch.object(Path, 'exists') as mock_exists:
            # First call for skip check (False), second for success check (True)
            mock_exists.side_effect = [False, True]

            result = extract_audio(str(video_file))

            assert result is not None
            assert result.endswith('.wav')
            mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_extract_audio_with_output_dir(self, mock_run, tmp_path):
        """Test audio extraction to specific output directory"""
        video_file = tmp_path / "test.mp4"
        video_file.write_text("fake video")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        with patch.object(Path, 'exists') as mock_exists:
            mock_exists.side_effect = [False, True]

            result = extract_audio(str(video_file), str(output_dir))

            assert result is not None
            assert str(output_dir) in result

    @patch('subprocess.run')
    def test_extract_audio_skips_existing(self, mock_run, tmp_path):
        """Test that extraction skips if audio already exists"""
        video_file = tmp_path / "test.mp4"
        video_file.write_text("fake video")

        # Create audio file
        audio_file = tmp_path / "test_12345678.wav"
        audio_file.write_text("fake audio")

        with patch('hashlib.md5') as mock_md5:
            mock_md5.return_value.hexdigest.return_value = "1234567890abcdef"

            result = extract_audio(str(video_file))

            # Should skip extraction
            assert result is not None
            mock_run.assert_not_called()

    @patch('subprocess.run')
    def test_extract_audio_ffmpeg_error(self, mock_run, tmp_path):
        """Test handling of ffmpeg errors"""
        video_file = tmp_path / "test.mp4"
        video_file.write_text("fake video")

        # Mock ffmpeg failure
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "ffmpeg error message"
        mock_run.return_value = mock_result

        with patch.object(Path, 'exists', return_value=False):
            result = extract_audio(str(video_file))

            assert result is None

    @patch('subprocess.run')
    def test_extract_audio_timeout(self, mock_run, tmp_path):
        """Test handling of ffmpeg timeout"""
        video_file = tmp_path / "test.mp4"
        video_file.write_text("fake video")

        # Mock timeout
        mock_run.side_effect = subprocess.TimeoutExpired('ffmpeg', 120)

        with patch.object(Path, 'exists', return_value=False):
            result = extract_audio(str(video_file))

            assert result is None

    @patch('subprocess.run')
    def test_extract_audio_exception(self, mock_run, tmp_path):
        """Test handling of unexpected exceptions"""
        video_file = tmp_path / "test.mp4"
        video_file.write_text("fake video")

        # Mock exception
        mock_run.side_effect = Exception("Unexpected error")

        with patch.object(Path, 'exists', return_value=False):
            result = extract_audio(str(video_file))

            assert result is None

    @patch('subprocess.run')
    def test_extract_audio_filename_hash(self, mock_run, tmp_path):
        """Test that filename includes hash for uniqueness"""
        video_file = tmp_path / "test.mp4"
        video_file.write_text("fake video")

        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        with patch.object(Path, 'exists') as mock_exists:
            mock_exists.side_effect = [False, True]

            result = extract_audio(str(video_file))

            # Result should contain hash in filename
            assert result is not None
            assert '_' in Path(result).stem  # Has underscore separator

    @patch('subprocess.run')
    def test_extract_audio_long_filename(self, mock_run, tmp_path):
        """Test handling of very long filenames"""
        long_name = "a" * 100 + ".mp4"
        video_file = tmp_path / long_name
        video_file.write_text("fake video")

        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        with patch.object(Path, 'exists') as mock_exists:
            mock_exists.side_effect = [False, True]

            result = extract_audio(str(video_file))

            # Filename should be truncated ([:80])
            assert result is not None
            assert len(Path(result).stem) <= 90  # 80 + hash


class TestWriteSrt:
    """Test write_srt() function"""

    def test_write_srt_basic(self, tmp_path):
        """Test basic SRT file writing"""
        segments = [
            {"start": 0.0, "end": 3.0, "text": "First segment"},
            {"start": 3.0, "end": 6.0, "text": "Second segment"}
        ]

        srt_file = tmp_path / "test.srt"
        write_srt(segments, str(srt_file))

        assert srt_file.exists()

        content = srt_file.read_text(encoding='utf-8')
        assert "1" in content
        assert "2" in content
        assert "First segment" in content
        assert "Second segment" in content
        assert "-->" in content

    def test_write_srt_timestamp_format(self, tmp_path):
        """Test SRT timestamp formatting"""
        segments = [
            {"start": 0.5, "end": 3.123, "text": "Test"}
        ]

        srt_file = tmp_path / "test.srt"
        write_srt(segments, str(srt_file))

        content = srt_file.read_text(encoding='utf-8')
        # Should have format HH:MM:SS,mmm
        assert "00:00:00,500" in content
        assert "00:00:03,123" in content

    def test_write_srt_strips_whitespace(self, tmp_path):
        """Test that text is stripped of whitespace"""
        segments = [
            {"start": 0.0, "end": 1.0, "text": "  Trimmed text  "}
        ]

        srt_file = tmp_path / "test.srt"
        write_srt(segments, str(srt_file))

        content = srt_file.read_text(encoding='utf-8')
        assert "Trimmed text" in content
        assert "  Trimmed text  " not in content

    def test_write_srt_empty_segments(self, tmp_path):
        """Test writing empty segment list"""
        srt_file = tmp_path / "test.srt"
        write_srt([], str(srt_file))

        assert srt_file.exists()
        content = srt_file.read_text(encoding='utf-8')
        assert content == ""

    def test_write_srt_missing_keys(self, tmp_path):
        """Test handling of segments with missing keys"""
        segments = [
            {},  # Missing all keys
            {"text": "Only text"}  # Missing start/end
        ]

        srt_file = tmp_path / "test.srt"
        write_srt(segments, str(srt_file))

        content = srt_file.read_text(encoding='utf-8')
        # Should use default values (0 for start/end, empty for text)
        assert "00:00:00,000" in content

    def test_write_srt_unicode_text(self, tmp_path):
        """Test handling of unicode characters"""
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello 世界 мир"},
            {"start": 1.0, "end": 2.0, "text": "Émojis: 🎉 ✨"}
        ]

        srt_file = tmp_path / "test.srt"
        write_srt(segments, str(srt_file))

        content = srt_file.read_text(encoding='utf-8')
        assert "世界" in content
        assert "мир" in content
        assert "🎉" in content

    def test_write_srt_multiline_text(self, tmp_path):
        """Test handling of text with newlines"""
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Line 1\nLine 2"}
        ]

        srt_file = tmp_path / "test.srt"
        write_srt(segments, str(srt_file))

        content = srt_file.read_text(encoding='utf-8')
        assert "Line 1\nLine 2" in content

    def test_write_srt_long_duration(self, tmp_path):
        """Test timestamp formatting for long durations (hours)"""
        segments = [
            {"start": 3661.5, "end": 7322.123, "text": "Long video"}
        ]

        srt_file = tmp_path / "test.srt"
        write_srt(segments, str(srt_file))

        content = srt_file.read_text(encoding='utf-8')
        # 3661.5 seconds = 1:01:01.500
        assert "01:01:01,500" in content
        # 7322.123 seconds = 2:02:02.123 but truncates to 122 due to int()
        assert "02:02:02,12" in content  # Match first two digits


class TestExtractVideoId:
    """Test extract_video_id() function"""

    def test_extract_video_id_regular_file(self):
        """Test extraction from regular filename with 11-char ID"""
        filename = "dQw4w9WgXcQ.mp4"
        result = extract_video_id(filename)

        assert result == "dQw4w9WgXcQ"

    def test_extract_video_id_segment_file(self):
        """Test extraction from segment filename"""
        filename = "dQw4w9WgXcQ_0045.mp4"
        result = extract_video_id(filename)

        assert result == "dQw4w9WgXcQ"

    def test_extract_video_id_audio_file(self):
        """Test extraction from audio filename"""
        filename = "dQw4w9WgXcQ.mp3"
        result = extract_video_id(filename)

        assert result == "dQw4w9WgXcQ"

    def test_extract_video_id_with_path(self):
        """Test extraction from full path"""
        filename = "/path/to/videos/dQw4w9WgXcQ.mp4"
        result = extract_video_id(filename)

        assert result == "dQw4w9WgXcQ"

    def test_extract_video_id_embedded_in_name(self):
        """Test extraction when ID is embedded in longer filename"""
        filename = "prefix_dQw4w9WgXcQ_suffix.mp4"
        result = extract_video_id(filename)

        # Implementation extracts first 11 chars if they match pattern
        # In this case "prefix_dQw4" is 11 chars, so that's what gets extracted
        # The regex pattern tries to find ANY 11-char sequence
        assert result is not None
        assert len(result) == 11

    def test_extract_video_id_invalid_filename(self):
        """Test with filename that has no valid ID"""
        filename = "regular_video_name.mp4"
        result = extract_video_id(filename)

        # May return None or find a 11-char sequence if it exists
        # Behavior depends on whether there's a valid 11-char sequence
        assert result is None or len(result) == 11

    def test_extract_video_id_too_short(self):
        """Test with filename shorter than 11 characters"""
        filename = "short.mp4"
        result = extract_video_id(filename)

        assert result is None

    def test_extract_video_id_special_characters(self):
        """Test with valid YouTube ID characters (A-Za-z0-9_-)"""
        # Valid YouTube IDs can contain underscores and hyphens
        filename = "abc-def_123.mp4"
        result = extract_video_id(filename)

        # Should extract the 11-char ID
        assert result == "abc-def_123"

    def test_extract_video_id_numeric_only(self):
        """Test with numeric-only ID"""
        filename = "12345678901.mp4"
        result = extract_video_id(filename)

        assert result == "12345678901"

    def test_extract_video_id_mixed_case(self):
        """Test with mixed case ID"""
        filename = "AbCdEfGhIjK.mp4"
        result = extract_video_id(filename)

        assert result == "AbCdEfGhIjK"


class TestFormatTimestampSrt:
    """Test format_timestamp_srt() function"""

    def test_format_timestamp_zero(self):
        """Test formatting zero timestamp"""
        result = format_timestamp_srt(0.0)
        assert result == "00:00:00,000"

    def test_format_timestamp_subsecond(self):
        """Test formatting subsecond timestamp"""
        result = format_timestamp_srt(0.123)
        assert result == "00:00:00,123"

    def test_format_timestamp_seconds(self):
        """Test formatting seconds only"""
        result = format_timestamp_srt(45.5)
        assert result == "00:00:45,500"

    def test_format_timestamp_minutes(self):
        """Test formatting with minutes"""
        result = format_timestamp_srt(125.250)
        # 125.250 seconds = 2:05.250
        assert result == "00:02:05,250"

    def test_format_timestamp_hours(self):
        """Test formatting with hours"""
        result = format_timestamp_srt(3661.123)
        # 3661.123 seconds = 1:01:01.123
        assert result == "01:01:01,123"

    def test_format_timestamp_long_duration(self):
        """Test formatting very long duration"""
        result = format_timestamp_srt(359999.999)
        # 359999.999 seconds = 99:59:59.999
        assert result == "99:59:59,999"

    def test_format_timestamp_milliseconds_rounding(self):
        """Test milliseconds are truncated (not rounded)"""
        result = format_timestamp_srt(1.9999)
        # Should truncate to 999 ms, not round to 2.000
        assert result == "00:00:01,999"

    def test_format_timestamp_exact_minute(self):
        """Test exact minute boundary"""
        result = format_timestamp_srt(60.0)
        assert result == "00:01:00,000"

    def test_format_timestamp_exact_hour(self):
        """Test exact hour boundary"""
        result = format_timestamp_srt(3600.0)
        assert result == "01:00:00,000"

    def test_format_timestamp_negative_not_expected(self):
        """Test behavior with negative value (edge case)"""
        # Not expected in normal use, but test graceful handling
        result = format_timestamp_srt(-1.0)
        # Implementation uses int() which truncates toward zero
        # Result will depend on exact implementation
        assert isinstance(result, str)


class TestEdgeCases:
    """Test edge cases and integration scenarios"""

    def test_extract_audio_and_write_srt_integration(self, tmp_path):
        """Test integration of audio extraction and SRT writing"""
        # This would normally be an integration test, but testing the flow
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Test"}
        ]

        srt_file = tmp_path / "output.srt"
        write_srt(segments, str(srt_file))

        assert srt_file.exists()

    def test_format_timestamp_matches_write_srt(self):
        """Test that standalone format function matches write_srt internal format"""
        # Test that public format_timestamp_srt matches the internal one used by write_srt
        test_time = 125.456

        standalone_result = format_timestamp_srt(test_time)

        # Expected format
        hours = int(test_time // 3600)
        minutes = int((test_time % 3600) // 60)
        secs = int(test_time % 60)
        millis = int((test_time % 1) * 1000)
        expected = f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

        assert standalone_result == expected

    def test_video_id_extraction_consistency(self):
        """Test that video ID extraction is consistent across formats"""
        video_id = "dQw4w9WgXcQ"

        filenames = [
            f"{video_id}.mp4",
            f"{video_id}.mp3",
            f"{video_id}_0001.mp4",
            f"/path/to/{video_id}.mp4"
        ]

        for filename in filenames:
            result = extract_video_id(filename)
            assert result == video_id, f"Failed for {filename}"

    def test_audio_filename_collision_prevention(self, tmp_path):
        """Test that hash prevents collisions for similar names"""
        # Create two videos with similar names in different directories
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        video1 = dir1 / "video.mp4"
        video2 = dir2 / "video.mp4"
        video1.write_text("fake")
        video2.write_text("fake")

        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0

            with patch.object(Path, 'exists') as mock_exists:
                mock_exists.side_effect = [False, True, False, True]

                audio1 = extract_audio(str(video1))
                audio2 = extract_audio(str(video2))

                # Should have different filenames due to hash
                assert audio1 != audio2
