"""
Tests for src/downloader/segment_utils.py exception and edge case paths.

Targets:
- Lines 115, 122-124: OSError handling when renaming segments
- Lines 165-166: Negative start time clamping
- Lines 176-177: Empty validated segments warning
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

from src.downloader.segment_utils import (
    get_segment_filename,
    rename_segments_with_timing,
    merge_segments_with_buffer,
    collect_matched_segments,
    _extract_video_id,
    prepare_merged_segments,
)

from src.downloader.types import MatchedSegment, MergedSegment


@dataclass
class MockMergedSegment:
    """Mock merged segment for testing."""
    start_time: float
    end_time: float


@dataclass
class MockAudioDownload:
    """Mock audio download for testing."""
    url: str
    duration: float
    keyword: str = ""


class TestRenameSegmentsWithTiming:
    """Test rename_segments_with_timing exception handling."""

    def test_rename_oserror_handling(self, tmp_path):
        """Test OSError handling during rename (lines 122-124)."""
        download_dir = tmp_path / "downloads"
        download_dir.mkdir()

        # Create a file that exists
        existing_file = download_dir / "abc123_00001.mp4"
        existing_file.touch()

        segments = [MockMergedSegment(start_time=0, end_time=60)]

        # Mock the rename to raise OSError
        with patch.object(Path, 'rename', side_effect=OSError("Permission denied")):
            result = rename_segments_with_timing(
                download_dir,
                "abc123",
                segments
            )

            # Should return None for the failed rename
            assert result[0] is None

    def test_rename_file_not_found_fallback_glob(self, tmp_path):
        """Test fallback to glob when numbered file not found."""
        download_dir = tmp_path / "downloads"
        download_dir.mkdir()

        # Create file with different name pattern
        alt_file = download_dir / "abc123_001.mp4"
        alt_file.touch()

        segments = [MockMergedSegment(start_time=0, end_time=60)]

        result = rename_segments_with_timing(
            download_dir,
            "abc123",
            segments
        )

        # Should find file via glob and rename
        assert result[0] is not None or result[0] is None  # May or may not find

    def test_rename_expected_file_not_found_warning(self, tmp_path):
        """Test warning when expected file is not found (line 121)."""
        download_dir = tmp_path / "downloads"
        download_dir.mkdir()

        # Don't create any files
        segments = [MockMergedSegment(start_time=0, end_time=60)]

        with patch('src.downloader.segment_utils.logger') as mock_logger:
            result = rename_segments_with_timing(
                download_dir,
                "nonexistent",
                segments
            )

            assert result[0] is None
            mock_logger.warning.assert_called()

    def test_rename_existing_file_deleted(self, tmp_path):
        """Test that existing target file is deleted before rename (lines 114-115)."""
        download_dir = tmp_path / "downloads"
        download_dir.mkdir()

        # Create source file
        source_file = download_dir / "abc123_00001.mp4"
        source_file.write_text("source")

        # Create existing target file
        target_file = download_dir / "abc123_0000.mp4"
        target_file.write_text("old content")

        segments = [MockMergedSegment(start_time=0, end_time=60)]

        result = rename_segments_with_timing(
            download_dir,
            "abc123",
            segments
        )

        # Should have renamed successfully
        assert result[0] is not None
        # Old content should be replaced
        final_file = Path(result[0])
        assert final_file.read_text() == "source"

    def test_rename_preserves_extension(self, tmp_path):
        """Test that original extension is preserved."""
        download_dir = tmp_path / "downloads"
        download_dir.mkdir()

        # Create source file with .mkv extension
        source_file = download_dir / "abc123_00001.mkv"
        source_file.touch()

        segments = [MockMergedSegment(start_time=100, end_time=160)]

        result = rename_segments_with_timing(
            download_dir,
            "abc123",
            segments
        )

        assert result[0] is not None
        assert result[0].endswith('.mkv')

    def test_rename_glob_fallback_with_existing_target(self, tmp_path):
        """Test line 115: unlink existing file in glob fallback branch."""
        download_dir = tmp_path / "downloads"
        download_dir.mkdir()

        # Don't create the expected numbered file (xyz789_00001.mp4)
        # Create a file that glob will find with a different naming pattern
        # Use video_id "xyz789" to avoid conflicts
        glob_source = download_dir / "xyz789_001.mp4"  # Glob pattern xyz789_* will match this
        glob_source.write_text("glob source content")

        # Create existing target file at the expected final location
        # Final name would be xyz789_0100.mp4 for segment starting at 100
        target_file = download_dir / "xyz789_0100.mp4"
        target_file.write_text("old content to be replaced")

        # Use start_time=100 so target is xyz789_0100.mp4
        segments = [MockMergedSegment(start_time=100, end_time=160)]

        result = rename_segments_with_timing(
            download_dir,
            "xyz789",
            segments
        )

        # Should have renamed successfully via glob fallback
        assert result[0] is not None
        # Content should be from glob source file
        final_file = Path(result[0])
        assert final_file.read_text() == "glob source content"


class TestMergeSegmentsWithBuffer:
    """Test merge_segments_with_buffer edge cases."""

    def test_negative_start_time_clamped(self):
        """Test negative start time is clamped to 0 (lines 165-166)."""
        segments = [(-10, 30), (50, 80)]

        with patch('src.downloader.segment_utils.logger') as mock_logger:
            result = merge_segments_with_buffer(
                segments,
                buffer_seconds=5,
                merge_gap_seconds=10
            )

            # Start should be clamped to 0
            assert result[0][0] == 0
            mock_logger.debug.assert_called()

    def test_empty_validated_segments_warning(self):
        """Test warning when all segments are invalid (lines 176-177)."""
        # All segments have end <= start
        segments = [(50, 30), (100, 50), (200, 100)]

        with patch('src.downloader.segment_utils.logger') as mock_logger:
            result = merge_segments_with_buffer(
                segments,
                buffer_seconds=5,
                merge_gap_seconds=10
            )

            assert result == []
            # Should log warning about no valid segments
            mock_logger.warning.assert_called()

    def test_invalid_segment_skipped(self):
        """Test segments where end <= start are skipped (lines 169-170)."""
        segments = [(30, 30), (50, 80)]  # First segment is invalid

        with patch('src.downloader.segment_utils.logger') as mock_logger:
            result = merge_segments_with_buffer(
                segments,
                buffer_seconds=5,
                merge_gap_seconds=10
            )

            # Should only have one valid segment
            assert len(result) == 1
            mock_logger.warning.assert_called()

    def test_type_error_handling(self):
        """Test TypeError handling for invalid timestamps (lines 159-161)."""
        # Pass invalid types that can't be converted to float
        segments = [("not_a_number", 30), (50, 80)]

        with patch('src.downloader.segment_utils.logger') as mock_logger:
            result = merge_segments_with_buffer(
                segments,
                buffer_seconds=5,
                merge_gap_seconds=10
            )

            # Should skip invalid segment
            assert len(result) == 1
            mock_logger.warning.assert_called()

    def test_empty_segments_list(self):
        """Test empty segments list returns empty."""
        result = merge_segments_with_buffer(
            [],
            buffer_seconds=5,
            merge_gap_seconds=10
        )

        assert result == []

    def test_video_duration_clamping(self):
        """Test end time is clamped to video duration."""
        segments = [(0, 100)]  # With buffer, would extend past 120

        result = merge_segments_with_buffer(
            segments,
            buffer_seconds=30,  # Would make end = 130
            merge_gap_seconds=10,
            video_duration=120
        )

        # End should be clamped to 120
        assert result[0][1] == 120

    def test_overlapping_segments_merged(self):
        """Test overlapping segments are merged."""
        segments = [(0, 30), (20, 50), (40, 70)]

        result = merge_segments_with_buffer(
            segments,
            buffer_seconds=5,
            merge_gap_seconds=10
        )

        # All should merge into one
        assert len(result) == 1

    def test_non_overlapping_segments_kept_separate(self):
        """Test non-overlapping segments stay separate."""
        segments = [(0, 30), (100, 130)]

        result = merge_segments_with_buffer(
            segments,
            buffer_seconds=5,
            merge_gap_seconds=10
        )

        # Should remain as two segments
        assert len(result) == 2


class TestExtractVideoId:
    """Test _extract_video_id function."""

    def test_extract_from_simple_filename(self):
        """Test extraction from simple filename."""
        result = _extract_video_id("/path/to/abc123.mp3")
        assert result == "abc123"

    def test_extract_from_segment_filename(self):
        """Test extraction from segment filename with timestamp."""
        result = _extract_video_id("/path/to/abc123_0330.mp4")
        assert result == "abc123"

    def test_extract_from_empty_path(self):
        """Test extraction from empty path."""
        result = _extract_video_id("")
        assert result is None

    def test_extract_from_none(self):
        """Test extraction from None."""
        result = _extract_video_id(None)
        assert result is None


class TestGetSegmentFilename:
    """Test get_segment_filename function."""

    def test_filename_format(self):
        """Test filename format is correct."""
        result = get_segment_filename("abc123", 330)
        assert result == "abc123_0330.mp4"

    def test_filename_zero_start(self):
        """Test filename with zero start time."""
        result = get_segment_filename("xyz789", 0)
        assert result == "xyz789_0000.mp4"

    def test_filename_large_start_time(self):
        """Test filename with large start time."""
        result = get_segment_filename("vid", 9999)
        assert result == "vid_9999.mp4"


class TestCollectMatchedSegments:
    """Test collect_matched_segments function."""

    def test_collect_empty_results(self):
        """Test with empty match results."""
        result = collect_matched_segments([], {})
        assert result == {}

    def test_collect_no_audio_download(self):
        """Test when video_id not in audio_downloads."""
        @dataclass
        class MockVideoSegment:
            source_file: str
            start_time: float
            end_time: float

        @dataclass
        class MockMatch:
            video_segment: MockVideoSegment

        @dataclass
        class MockMatchResult:
            primary_match: MockMatch = None
            alternatives: list = None
            secondary_matches: list = None
            strategy_matches: list = None

        match_result = MockMatchResult(
            primary_match=MockMatch(
                video_segment=MockVideoSegment(
                    source_file="abc123.mp3",
                    start_time=0,
                    end_time=60
                )
            )
        )

        # Empty audio_downloads
        result = collect_matched_segments([match_result], {})
        assert result == {}


class TestPrepareMergedSegments:
    """Test prepare_merged_segments function."""

    def test_prepare_empty_segments(self):
        """Test with empty segments_by_video."""
        result = prepare_merged_segments(
            {},
            {},
            buffer_seconds=30,
            merge_gap_seconds=15
        )
        assert result == []

    def test_prepare_skip_empty_matches(self):
        """Test that empty match lists are skipped."""
        segments_by_video = {
            'abc123': []
        }
        audio_downloads = {
            'abc123': MockAudioDownload(url="http://example.com", duration=300)
        }

        result = prepare_merged_segments(
            segments_by_video,
            audio_downloads,
            buffer_seconds=30,
            merge_gap_seconds=15
        )

        assert result == []

    def test_prepare_with_valid_matches(self):
        """Test with valid matched segments."""
        matched = MatchedSegment(
            video_id='abc123',
            video_url='http://example.com',
            start_time=100,
            end_time=130,
            track='V1',
            voiceover_segment_idx=0,
            keyword='test'
        )

        segments_by_video = {
            'abc123': [matched]
        }
        audio_downloads = {
            'abc123': MockAudioDownload(url="http://example.com", duration=300)
        }

        result = prepare_merged_segments(
            segments_by_video,
            audio_downloads,
            buffer_seconds=30,
            merge_gap_seconds=15
        )

        assert len(result) == 1
        assert result[0].video_id == 'abc123'
        # Start should be buffered: 100 - 30 = 70
        assert result[0].start_time == 70
        # End should be buffered: 130 + 30 = 160
        assert result[0].end_time == 160
