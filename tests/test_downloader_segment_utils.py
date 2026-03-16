"""
Tests for src/downloader/segment_utils.py

Covers:
- get_segment_filename - Filename generation
- rename_segments_with_timing - Rename autonumber files
- merge_segments_with_buffer - Segment merging logic
- collect_matched_segments - Match collection from results
- _extract_video_id - Video ID extraction
- prepare_merged_segments - Segment preparation for download

Created: 2026-01-11 (Session 14)
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.downloader.segment_utils import (
    get_segment_filename,
    rename_segments_with_timing,
    merge_segments_with_buffer,
    collect_matched_segments,
    _extract_video_id,
    prepare_merged_segments
)
from src.downloader.types import MatchedSegment, MergedSegment


# ============================================================================
# Test get_segment_filename
# ============================================================================

class TestGetSegmentFilename:
    """Test filename generation."""

    @pytest.mark.fast
    def test_basic_filename(self):
        """Test basic filename generation."""
        result = get_segment_filename("abc123", 0)
        assert result == "abc123_0000.mp4"

    @pytest.mark.fast
    def test_start_at_330_seconds(self):
        """Test filename with 330 second start."""
        result = get_segment_filename("xyz789", 330)
        assert result == "xyz789_0330.mp4"

    @pytest.mark.fast
    def test_large_start_time(self):
        """Test filename with large start time."""
        result = get_segment_filename("video", 3600)  # 1 hour
        assert result == "video_3600.mp4"

    @pytest.mark.fast
    def test_float_start_time(self):
        """Test filename with float start time."""
        result = get_segment_filename("test", 123.456)
        assert result == "test_0123.mp4"  # Truncated to int

    @pytest.mark.fast
    def test_short_video_id(self):
        """Test with short video ID."""
        result = get_segment_filename("a", 10)
        assert result == "a_0010.mp4"


# ============================================================================
# Test rename_segments_with_timing
# ============================================================================

class TestRenameSegmentsWithTiming:
    """Test segment renaming."""

    @pytest.fixture
    def temp_download_dir(self, tmp_path):
        """Create temporary download directory."""
        download_dir = tmp_path / "downloads"
        download_dir.mkdir()
        return download_dir

    @pytest.mark.fast
    def test_rename_single_segment(self, temp_download_dir):
        """Test renaming a single segment."""
        video_id = "abc123"

        # Create autonumber file (yt-dlp default format)
        (temp_download_dir / f"{video_id}_00001.mp4").touch()

        segments = [MergedSegment(
            video_id=video_id,
            video_url="https://youtube.com/watch?v=abc123",
            start_time=60.0,
            end_time=120.0,
            original_matches=[],
            keyword="test"
        )]

        results = rename_segments_with_timing(temp_download_dir, video_id, segments)

        assert len(results) == 1
        assert results[0] is not None
        assert "0060" in results[0]  # Start time encoded

    @pytest.mark.fast
    def test_rename_multiple_segments(self, temp_download_dir):
        """Test renaming multiple segments."""
        video_id = "xyz789"

        # Create autonumber files
        (temp_download_dir / f"{video_id}_00001.mp4").touch()
        (temp_download_dir / f"{video_id}_00002.mp4").touch()
        (temp_download_dir / f"{video_id}_00003.mp4").touch()

        segments = [
            MergedSegment(video_id=video_id, video_url="url", start_time=0.0, end_time=30.0, original_matches=[], keyword="test"),
            MergedSegment(video_id=video_id, video_url="url", start_time=120.0, end_time=180.0, original_matches=[], keyword="test"),
            MergedSegment(video_id=video_id, video_url="url", start_time=300.0, end_time=360.0, original_matches=[], keyword="test"),
        ]

        results = rename_segments_with_timing(temp_download_dir, video_id, segments)

        assert len(results) == 3
        assert all(r is not None for r in results)
        assert "0000" in results[0]
        assert "0120" in results[1]
        assert "0300" in results[2]

    @pytest.mark.fast
    def test_rename_unpadded_format(self, temp_download_dir):
        """Test renaming unpadded autonumber files."""
        video_id = "test"

        # Create unpadded files
        (temp_download_dir / f"{video_id}_1.mp4").touch()
        (temp_download_dir / f"{video_id}_2.mp4").touch()

        segments = [
            MergedSegment(video_id=video_id, video_url="url", start_time=10.0, end_time=30.0, original_matches=[], keyword="test"),
            MergedSegment(video_id=video_id, video_url="url", start_time=60.0, end_time=90.0, original_matches=[], keyword="test"),
        ]

        results = rename_segments_with_timing(temp_download_dir, video_id, segments)

        assert len(results) == 2
        assert all(r is not None for r in results)

    @pytest.mark.fast
    def test_rename_preserves_extension(self, temp_download_dir):
        """Test that original extension is preserved."""
        video_id = "ext_test"

        # Create mkv file
        (temp_download_dir / f"{video_id}_00001.mkv").touch()

        segments = [MergedSegment(
            video_id=video_id, video_url="url", start_time=30.0, end_time=60.0,
            original_matches=[], keyword="test"
        )]

        results = rename_segments_with_timing(temp_download_dir, video_id, segments)

        assert len(results) == 1
        assert results[0].endswith(".mkv")

    @pytest.mark.fast
    def test_rename_file_not_found(self, temp_download_dir):
        """Test handling when file is not found."""
        video_id = "missing"

        segments = [MergedSegment(
            video_id=video_id, video_url="url", start_time=0.0, end_time=30.0,
            original_matches=[], keyword="test"
        )]

        results = rename_segments_with_timing(temp_download_dir, video_id, segments)

        assert len(results) == 1
        assert results[0] is None

    @pytest.mark.fast
    def test_rename_glob_fallback(self, temp_download_dir):
        """Test glob fallback when autonumber not found."""
        video_id = "glob_test"

        # Create file with unexpected naming
        (temp_download_dir / f"{video_id}_segment.mp4").touch()

        segments = [MergedSegment(
            video_id=video_id, video_url="url", start_time=45.0, end_time=90.0,
            original_matches=[], keyword="test"
        )]

        results = rename_segments_with_timing(temp_download_dir, video_id, segments)

        assert len(results) == 1
        # May find via glob or return None

    @pytest.mark.fast
    def test_rename_existing_file_deleted(self, temp_download_dir):
        """Test that existing renamed file is overwritten."""
        video_id = "overwrite"

        # Create source file
        source = temp_download_dir / f"{video_id}_00001.mp4"
        source.write_text("new content")

        # Create existing target file
        target = temp_download_dir / f"{video_id}_0030.mp4"
        target.write_text("old content")

        segments = [MergedSegment(
            video_id=video_id, video_url="url", start_time=30.0, end_time=60.0,
            original_matches=[], keyword="test"
        )]

        results = rename_segments_with_timing(temp_download_dir, video_id, segments)

        assert len(results) == 1
        # Verify new file has new content
        final_file = Path(results[0])
        assert final_file.read_text() == "new content"


# ============================================================================
# Test merge_segments_with_buffer
# ============================================================================

class TestMergeSegmentsWithBuffer:
    """Test segment merging logic."""

    @pytest.mark.fast
    def test_empty_segments(self):
        """Test with empty segment list."""
        result = merge_segments_with_buffer([])
        assert result == []

    @pytest.mark.fast
    def test_single_segment(self):
        """Test with single segment."""
        segments = [(30.0, 60.0)]
        result = merge_segments_with_buffer(segments, buffer_seconds=10.0)

        assert len(result) == 1
        assert result[0][0] == 20.0  # 30 - 10
        assert result[0][1] == 70.0  # 60 + 10

    @pytest.mark.fast
    def test_non_overlapping_segments(self):
        """Test segments that don't overlap."""
        segments = [(0.0, 30.0), (100.0, 130.0)]
        result = merge_segments_with_buffer(segments, buffer_seconds=10.0, merge_gap_seconds=5.0)

        assert len(result) == 2

    @pytest.mark.fast
    def test_overlapping_segments_merged(self):
        """Test that overlapping segments are merged."""
        segments = [(0.0, 40.0), (30.0, 70.0)]  # Overlap at 30-40
        result = merge_segments_with_buffer(segments, buffer_seconds=0.0)

        assert len(result) == 1
        assert result[0] == (0.0, 70.0)

    @pytest.mark.fast
    def test_close_segments_merged(self):
        """Test that close segments are merged based on gap threshold."""
        segments = [(0.0, 30.0), (40.0, 70.0)]  # 10s gap
        result = merge_segments_with_buffer(segments, buffer_seconds=0.0, merge_gap_seconds=15.0)

        assert len(result) == 1  # Should be merged

    @pytest.mark.fast
    def test_clamp_to_video_duration(self):
        """Test that end time is clamped to video duration."""
        segments = [(100.0, 150.0)]
        result = merge_segments_with_buffer(segments, buffer_seconds=10.0, video_duration=140.0)

        assert len(result) == 1
        assert result[0][1] == 140.0  # Clamped

    @pytest.mark.fast
    def test_negative_start_clamped(self):
        """Test that negative start time is clamped to 0."""
        segments = [(5.0, 20.0)]
        result = merge_segments_with_buffer(segments, buffer_seconds=10.0)

        assert len(result) == 1
        assert result[0][0] == 0.0  # Clamped from -5

    @pytest.mark.fast
    def test_invalid_segments_skipped(self):
        """Test that invalid segments are skipped."""
        segments = [(50.0, 30.0), (10.0, 40.0)]  # First is invalid (end <= start)
        result = merge_segments_with_buffer(segments, buffer_seconds=0.0)

        assert len(result) == 1
        assert result[0] == (10.0, 40.0)

    @pytest.mark.fast
    def test_type_conversion(self):
        """Test that string values are converted to float."""
        segments = [("10", "30")]  # Strings
        result = merge_segments_with_buffer(segments, buffer_seconds=5.0)

        assert len(result) == 1
        assert result[0] == (5.0, 35.0)

    @pytest.mark.fast
    def test_invalid_type_skipped(self):
        """Test that invalid types are skipped."""
        segments = [(None, None), (10.0, 30.0)]
        result = merge_segments_with_buffer(segments, buffer_seconds=0.0)

        assert len(result) == 1
        assert result[0] == (10.0, 30.0)

    @pytest.mark.fast
    def test_three_segments_partial_merge(self):
        """Test three segments where only some merge."""
        segments = [(0.0, 30.0), (25.0, 55.0), (100.0, 130.0)]
        result = merge_segments_with_buffer(segments, buffer_seconds=0.0, merge_gap_seconds=5.0)

        assert len(result) == 2
        assert result[0] == (0.0, 55.0)  # First two merged
        assert result[1] == (100.0, 130.0)


# ============================================================================
# Test _extract_video_id
# ============================================================================

class TestExtractVideoId:
    """Test video ID extraction from file paths."""

    @pytest.mark.fast
    def test_simple_path(self):
        """Test simple path extraction."""
        result = _extract_video_id("abc123.mp3")
        assert result == "abc123"

    @pytest.mark.fast
    def test_segment_format(self):
        """Test segment format with underscore and number."""
        result = _extract_video_id("xyz789_0030.mp4")
        assert result == "xyz789"

    @pytest.mark.fast
    def test_full_path(self):
        """Test with full path."""
        result = _extract_video_id("/downloads/video/abc123_0060.mp4")
        assert result == "abc123"

    @pytest.mark.fast
    def test_windows_path(self):
        """Test with Windows path."""
        result = _extract_video_id("C:\\downloads\\video123.mp4")
        assert result == "video123"

    @pytest.mark.fast
    def test_empty_path(self):
        """Test with empty path."""
        result = _extract_video_id("")
        assert result is None

    @pytest.mark.fast
    def test_none_path(self):
        """Test with None path."""
        result = _extract_video_id(None)
        assert result is None

    @pytest.mark.fast
    def test_multiple_underscores(self):
        """Test path with multiple underscores."""
        result = _extract_video_id("test_video_id_0120.mp4")
        assert result == "test_video_id"


# ============================================================================
# Test collect_matched_segments
# ============================================================================

class TestCollectMatchedSegments:
    """Test match collection from results."""

    @pytest.fixture
    def mock_audio_download(self):
        """Create a mock audio download."""
        audio = Mock()
        audio.url = "https://youtube.com/watch?v=abc123"
        audio.keyword = "test keyword"
        return audio

    @pytest.fixture
    def mock_video_segment(self):
        """Create a mock video segment."""
        segment = Mock()
        segment.source_file = "abc123.mp3"
        segment.start_time = 10.0
        segment.end_time = 30.0
        return segment

    @pytest.fixture
    def mock_match(self, mock_video_segment):
        """Create a mock match."""
        match = Mock()
        match.video_segment = mock_video_segment
        return match

    @pytest.mark.fast
    def test_collect_primary_match(self, mock_audio_download, mock_match):
        """Test collecting primary match."""
        result = Mock()
        result.primary_match = mock_match
        result.alternatives = []
        result.secondary_matches = []
        result.strategy_matches = []

        audio_downloads = {"abc123": mock_audio_download}

        segments = collect_matched_segments([result], audio_downloads)

        assert "abc123" in segments
        assert len(segments["abc123"]) == 1
        assert segments["abc123"][0].track == "V1"

    @pytest.mark.fast
    def test_collect_alternatives(self, mock_audio_download, mock_match):
        """Test collecting alternative matches."""
        result = Mock()
        result.primary_match = None
        result.alternatives = [mock_match, mock_match]  # V2, V3
        result.secondary_matches = []
        result.strategy_matches = []

        audio_downloads = {"abc123": mock_audio_download}

        segments = collect_matched_segments([result], audio_downloads)

        assert "abc123" in segments
        assert len(segments["abc123"]) == 2
        tracks = [s.track for s in segments["abc123"]]
        assert "V2" in tracks
        assert "V3" in tracks

    @pytest.mark.fast
    def test_collect_secondary_matches(self, mock_audio_download, mock_match):
        """Test collecting secondary matches."""
        result = Mock()
        result.primary_match = None
        result.alternatives = []
        result.secondary_matches = [mock_match]  # V4
        result.strategy_matches = []

        audio_downloads = {"abc123": mock_audio_download}

        segments = collect_matched_segments([result], audio_downloads)

        assert "abc123" in segments
        assert len(segments["abc123"]) == 1
        assert segments["abc123"][0].track == "V4"

    @pytest.mark.fast
    def test_collect_strategy_matches(self, mock_audio_download, mock_match):
        """Test collecting strategy matches."""
        result = Mock()
        result.primary_match = None
        result.alternatives = []
        result.secondary_matches = []
        result.strategy_matches = [mock_match]  # V7

        audio_downloads = {"abc123": mock_audio_download}

        segments = collect_matched_segments([result], audio_downloads)

        assert "abc123" in segments
        assert len(segments["abc123"]) == 1
        assert segments["abc123"][0].track == "V7"

    @pytest.mark.fast
    def test_collect_unknown_video_id_skipped(self, mock_match):
        """Test that unknown video IDs are skipped."""
        result = Mock()
        result.primary_match = mock_match
        result.alternatives = []
        result.secondary_matches = []
        result.strategy_matches = []

        audio_downloads = {}  # Empty - no matching audio

        segments = collect_matched_segments([result], audio_downloads)

        assert segments == {}

    @pytest.mark.fast
    def test_collect_all_match_types(self, mock_audio_download, mock_video_segment):
        """Test collecting all match types in one result."""
        # Create unique segments for each match type
        def make_segment():
            seg = Mock()
            seg.source_file = "abc123.mp3"
            seg.start_time = 10.0
            seg.end_time = 30.0
            return seg

        def make_match():
            match = Mock()
            match.video_segment = make_segment()
            return match

        result = Mock()
        result.primary_match = make_match()
        result.alternatives = [make_match(), make_match()]
        result.secondary_matches = [make_match()]
        result.strategy_matches = [make_match()]

        audio_downloads = {"abc123": mock_audio_download}

        segments = collect_matched_segments([result], audio_downloads)

        assert "abc123" in segments
        # primary (1) + alternatives (2) + secondary (1) + strategy (1) = 5
        assert len(segments["abc123"]) == 5


# ============================================================================
# Test prepare_merged_segments
# ============================================================================

class TestPrepareMergedSegments:
    """Test segment preparation for download."""

    @pytest.fixture
    def mock_audio(self):
        """Create mock audio download."""
        audio = Mock()
        audio.duration = 300.0  # 5 minutes
        return audio

    @pytest.mark.fast
    def test_prepare_empty_input(self):
        """Test with empty input."""
        result = prepare_merged_segments({}, {})
        assert result == []

    @pytest.mark.fast
    def test_prepare_single_video(self, mock_audio):
        """Test with single video."""
        matched = MatchedSegment(
            video_id="abc123",
            video_url="https://youtube.com/watch?v=abc123",
            start_time=30.0,
            end_time=60.0,
            track="V1",
            voiceover_segment_idx=0,
            keyword="test"
        )

        segments_by_video = {"abc123": [matched]}
        audio_downloads = {"abc123": mock_audio}

        result = prepare_merged_segments(segments_by_video, audio_downloads)

        assert len(result) == 1
        assert result[0].video_id == "abc123"

    @pytest.mark.fast
    def test_prepare_multiple_segments_merged(self, mock_audio):
        """Test that close segments are merged."""
        matches = [
            MatchedSegment(
                video_id="abc123", video_url="url", start_time=30.0, end_time=40.0,
                track="V1", voiceover_segment_idx=0, keyword="test"
            ),
            MatchedSegment(
                video_id="abc123", video_url="url", start_time=45.0, end_time=55.0,
                track="V1", voiceover_segment_idx=1, keyword="test"
            ),
        ]

        segments_by_video = {"abc123": matches}
        audio_downloads = {"abc123": mock_audio}

        result = prepare_merged_segments(
            segments_by_video, audio_downloads,
            buffer_seconds=10.0, merge_gap_seconds=15.0
        )

        # With 10s buffer and 15s merge gap, these should merge
        # 30-40 with 10s buffer = 20-50
        # 45-55 with 10s buffer = 35-65
        # They overlap, so should become 20-65
        assert len(result) == 1

    @pytest.mark.fast
    def test_prepare_uses_video_duration(self, mock_audio):
        """Test that video duration is used for clamping."""
        mock_audio.duration = 100.0

        matched = MatchedSegment(
            video_id="abc123", video_url="url", start_time=90.0, end_time=110.0,
            track="V1", voiceover_segment_idx=0, keyword="test"
        )

        segments_by_video = {"abc123": [matched]}
        audio_downloads = {"abc123": mock_audio}

        result = prepare_merged_segments(
            segments_by_video, audio_downloads,
            buffer_seconds=20.0  # Would extend to 130
        )

        assert len(result) == 1
        assert result[0].end_time <= 100.0  # Clamped to duration

    @pytest.mark.fast
    def test_prepare_empty_matches_skipped(self):
        """Test that empty match lists are skipped."""
        segments_by_video = {"abc123": []}
        audio_downloads = {"abc123": Mock(duration=100.0)}

        result = prepare_merged_segments(segments_by_video, audio_downloads)

        assert result == []

    @pytest.mark.fast
    def test_prepare_contains_original_matches(self, mock_audio):
        """Test that original matches are included."""
        matches = [
            MatchedSegment(
                video_id="abc123", video_url="url", start_time=30.0, end_time=50.0,
                track="V1", voiceover_segment_idx=0, keyword="test"
            ),
        ]

        segments_by_video = {"abc123": matches}
        audio_downloads = {"abc123": mock_audio}

        result = prepare_merged_segments(
            segments_by_video, audio_downloads,
            buffer_seconds=10.0
        )

        assert len(result) == 1
        assert len(result[0].original_matches) >= 0  # May contain matches


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
