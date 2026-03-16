"""
Test Suite for SegmentPathIndex

Tests the O(1) segment lookup index.
"""

import pytest
from src.stages.output import SegmentPathIndex, SegmentInfo


class TestSegmentPathIndex:
    """Tests for SegmentPathIndex class."""

    def test_basic_lookup(self):
        """Test basic (video_id, start, end) lookup."""
        segments = [
            SegmentInfo(video_id="abc123", file="/path/video1.mp4", original_start=0.0, original_end=120.0),
            SegmentInfo(video_id="abc123", file="/path/video2.mp4", original_start=120.0, original_end=240.0),
            SegmentInfo(video_id="def456", file="/path/video3.mp4", original_start=0.0, original_end=60.0),
        ]
        index = SegmentPathIndex(segments)

        # Test exact lookups
        assert index.lookup("abc123", 0, 120) == "/path/video1.mp4"
        assert index.lookup("abc123", 120, 240) == "/path/video2.mp4"
        assert index.lookup("def456", 0, 60) == "/path/video3.mp4"

        # Test non-existent lookup
        assert index.lookup("xyz999", 0, 60) is None

    def test_lookup_with_float_times(self):
        """Test lookup works with float times (converted to int)."""
        segments = [
            SegmentInfo(video_id="abc123", file="/path/video1.mp4", original_start=0.5, original_end=120.3),
        ]
        index = SegmentPathIndex(segments)

        # Float times should be converted to int for lookup
        assert index.lookup("abc123", 0.5, 120.3) == "/path/video1.mp4"
        # 0.5 -> 0 and 120.3 -> 120 when stored, so lookup with 0 and 120 works
        assert index.lookup("abc123", 0, 120) == "/path/video1.mp4"

    def test_flat_dir_precedence(self):
        """Test flat dir takes precedence over legacy when segment exists in both."""
        # Legacy segment
        legacy_seg = SegmentInfo(
            video_id="abc123", file="/legacy/path.mp4",
            original_start=0.0, original_end=120.0
        )
        # Flat dir segment (same video_id, start, end)
        flat_seg = SegmentInfo(
            video_id="abc123", file="/flat/path.mp4",
            original_start=0.0, original_end=120.0
        )

        # Both in list - flat should take precedence
        index = SegmentPathIndex([legacy_seg, flat_seg])

        assert index.lookup("abc123", 0, 120) == "/flat/path.mp4"

    def test_has_video(self):
        """Test has_video method."""
        segments = [
            SegmentInfo(video_id="abc123", file="/path/video1.mp4", original_start=0.0, original_end=120.0),
            SegmentInfo(video_id="def456", file="/path/video2.mp4", original_start=0.0, original_end=60.0),
        ]
        index = SegmentPathIndex(segments)

        assert index.has_video("abc123") is True
        assert index.has_video("def456") is True
        assert index.has_video("xyz999") is False

    def test_get_by_video_id(self):
        """Test get_by_video_id returns all segments for a video."""
        segments = [
            SegmentInfo(video_id="abc123", file="/path/video1.mp4", original_start=0.0, original_end=120.0),
            SegmentInfo(video_id="abc123", file="/path/video2.mp4", original_start=120.0, original_end=240.0),
            SegmentInfo(video_id="def456", file="/path/video3.mp4", original_start=0.0, original_end=60.0),
        ]
        index = SegmentPathIndex(segments)

        abc_segments = index.get_by_video_id("abc123")
        assert len(abc_segments) == 2
        assert (0, 120, "/path/video1.mp4") in abc_segments
        assert (120, 240, "/path/video2.mp4") in abc_segments

        def_segments = index.get_by_video_id("def456")
        assert len(def_segments) == 1

        xyz_segments = index.get_by_video_id("xyz999")
        assert len(xyz_segments) == 0

    def test_len_and_bool(self):
        """Test __len__ and __bool__ methods."""
        # Empty index
        empty_index = SegmentPathIndex([])
        assert len(empty_index) == 0
        assert bool(empty_index) is False

        # Non-empty index
        segments = [
            SegmentInfo(video_id="abc123", file="/path/video1.mp4", original_start=0.0, original_end=120.0),
        ]
        index = SegmentPathIndex(segments)
        assert len(index) == 1
        assert bool(index) is True

    def test_iteration(self):
        """Test iteration over segments."""
        segments = [
            SegmentInfo(video_id="abc123", file="/path/video1.mp4", original_start=0.0, original_end=120.0),
            SegmentInfo(video_id="def456", file="/path/video2.mp4", original_start=0.0, original_end=60.0),
        ]
        index = SegmentPathIndex(segments)

        # Can iterate like a list
        iterated = list(index)
        assert len(iterated) == 2
        assert iterated[0].video_id == "abc123"
        assert iterated[1].video_id == "def456"

    def test_segments_property(self):
        """Test segments property returns original list."""
        segments = [
            SegmentInfo(video_id="abc123", file="/path/video1.mp4", original_start=0.0, original_end=120.0),
        ]
        index = SegmentPathIndex(segments)

        # segments property should return the list
        assert index.segments == segments
        assert len(index.segments) == 1
        assert index.segments[0].video_id == "abc123"


class TestSegmentPathIndexIntegration:
    """Integration tests for SegmentPathIndex with OUTPUT stage."""

    def test_index_lookup_for_known_segment(self):
        """Test index lookup returns correct path for known segment file."""
        # Simulate a known segment file
        segments = [
            SegmentInfo(
                video_id="dQw4w9WgXcQ",
                file="E:/videos/dQw4w9WgXcQ_0_120.mp4",
                original_start=0.0,
                original_end=120.0
            ),
        ]
        index = SegmentPathIndex(segments)

        # Verify lookup works
        result = index.lookup("dQw4w9WgXcQ", 0, 120)
        assert result == "E:/videos/dQw4w9WgXcQ_0_120.mp4"

    def test_index_handles_legacy_and_flat_dirs(self):
        """Test index handles segments in both legacy and flat directories."""
        # Legacy format: {video_id}_{start_4digit}.mp4
        legacy_seg = SegmentInfo(
            video_id="abc123",
            file="E:/v/old_project_segments/abc123_0000.mp4",
            original_start=0.0,
            original_end=120.0
        )

        # Flat format: {video_id}_{start}_{end}.mp4
        flat_seg = SegmentInfo(
            video_id="abc123",
            file="E:/v/new_project/abc123_0_120.mp4",
            original_start=0.0,
            original_end=120.0
        )

        # Both present - flat should win
        index = SegmentPathIndex([legacy_seg, flat_seg])
        result = index.lookup("abc123", 0, 120)

        assert result == "E:/v/new_project/abc123_0_120.mp4"
