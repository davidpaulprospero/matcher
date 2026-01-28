"""
Tests for src/matching/tracking.py to achieve 100% coverage.

Targets:
- TimelineVarietyTracker: binary search, window filtering, recording
- GlobalClipTracker: clip ID generation, audio-first pattern, usage tracking
"""

import pytest
from unittest.mock import MagicMock, patch

from src.matching.tracking import TimelineVarietyTracker, GlobalClipTracker
from src.utils import SRTSegment


class TestTimelineVarietyTracker:
    """Test TimelineVarietyTracker class."""

    @pytest.mark.fast
    def test_init_default_values(self):
        """Test initialization with default values."""
        tracker = TimelineVarietyTracker()
        assert tracker.timeline_window == 600.0
        assert tracker.max_repeats == 1
        assert len(tracker.track_usage) == 0

    @pytest.mark.fast
    def test_init_custom_values(self):
        """Test initialization with custom values."""
        tracker = TimelineVarietyTracker(timeline_window=300.0, max_repeats=2)
        assert tracker.timeline_window == 300.0
        assert tracker.max_repeats == 2

    @pytest.mark.fast
    def test_get_excluded_sources_empty(self):
        """Test get_excluded_sources on empty tracker."""
        tracker = TimelineVarietyTracker()
        excluded = tracker.get_excluded_sources("V1", 100.0)
        assert excluded == set()

    @pytest.mark.fast
    def test_record_usage_append(self):
        """Test recording usage at end (most common case)."""
        tracker = TimelineVarietyTracker()

        tracker.record_usage("V1", "/video/a.mp4", 0.0)
        tracker.record_usage("V1", "/video/b.mp4", 10.0)
        tracker.record_usage("V1", "/video/c.mp4", 20.0)

        assert len(tracker.track_usage["V1"]) == 3
        assert tracker.track_usage["V1"][0] == (0.0, "/video/a.mp4")
        assert tracker.track_usage["V1"][1] == (10.0, "/video/b.mp4")
        assert tracker.track_usage["V1"][2] == (20.0, "/video/c.mp4")

    @pytest.mark.fast
    def test_record_usage_insert_middle(self):
        """Test recording usage that needs to be inserted (not appended)."""
        tracker = TimelineVarietyTracker()

        # Record in non-sequential order to trigger binary search insertion
        tracker.record_usage("V1", "/video/a.mp4", 0.0)
        tracker.record_usage("V1", "/video/c.mp4", 20.0)
        # Insert in the middle - triggers lines 106-113
        tracker.record_usage("V1", "/video/b.mp4", 10.0)

        assert len(tracker.track_usage["V1"]) == 3
        # Should be sorted by position
        assert tracker.track_usage["V1"][0] == (0.0, "/video/a.mp4")
        assert tracker.track_usage["V1"][1] == (10.0, "/video/b.mp4")
        assert tracker.track_usage["V1"][2] == (20.0, "/video/c.mp4")

    @pytest.mark.fast
    def test_record_usage_insert_beginning(self):
        """Test recording usage at the beginning."""
        tracker = TimelineVarietyTracker()

        tracker.record_usage("V1", "/video/b.mp4", 10.0)
        tracker.record_usage("V1", "/video/c.mp4", 20.0)
        # Insert at beginning
        tracker.record_usage("V1", "/video/a.mp4", 5.0)

        assert len(tracker.track_usage["V1"]) == 3
        assert tracker.track_usage["V1"][0] == (5.0, "/video/a.mp4")

    @pytest.mark.fast
    def test_get_excluded_sources_single_source_at_max(self):
        """Test exclusion when source reaches max_repeats."""
        tracker = TimelineVarietyTracker(timeline_window=600.0, max_repeats=1)

        # Add source at position 0
        tracker.record_usage("V1", "/video/a.mp4", 0.0)

        # Check exclusion at position 100 (within window)
        excluded = tracker.get_excluded_sources("V1", 100.0)
        assert "/video/a.mp4" in excluded

    @pytest.mark.fast
    def test_get_excluded_sources_within_window(self):
        """Test that sources within window are excluded."""
        tracker = TimelineVarietyTracker(timeline_window=100.0, max_repeats=2)

        # Add same source twice
        tracker.record_usage("V1", "/video/a.mp4", 0.0)
        tracker.record_usage("V1", "/video/a.mp4", 50.0)

        # At position 80, both usages are in window [0, 80)
        # But we're at 80, so window is [-20, 80) which only contains 0 and 50
        excluded = tracker.get_excluded_sources("V1", 80.0)
        assert "/video/a.mp4" in excluded

    @pytest.mark.fast
    def test_get_excluded_sources_outside_window(self):
        """Test that sources outside window are not excluded."""
        tracker = TimelineVarietyTracker(timeline_window=100.0, max_repeats=1)

        # Add source at position 0
        tracker.record_usage("V1", "/video/a.mp4", 0.0)

        # At position 200, the window is [100, 200) - source at 0 is outside
        excluded = tracker.get_excluded_sources("V1", 200.0)
        assert "/video/a.mp4" not in excluded

    @pytest.mark.fast
    def test_get_excluded_sources_binary_search_left_branch(self):
        """Test binary search takes left branch (position < window_start)."""
        tracker = TimelineVarietyTracker(timeline_window=50.0, max_repeats=1)

        # Add sources at various positions
        tracker.record_usage("V1", "/video/a.mp4", 0.0)
        tracker.record_usage("V1", "/video/b.mp4", 30.0)
        tracker.record_usage("V1", "/video/c.mp4", 60.0)
        tracker.record_usage("V1", "/video/d.mp4", 90.0)

        # At position 100, window is [50, 100)
        # Binary search should find that positions 0 and 30 are before window
        excluded = tracker.get_excluded_sources("V1", 100.0)

        # Only sources in [50, 100) should be excluded
        assert "/video/a.mp4" not in excluded  # At 0, before window
        assert "/video/b.mp4" not in excluded  # At 30, before window
        assert "/video/c.mp4" in excluded      # At 60, in window
        assert "/video/d.mp4" in excluded      # At 90, in window

    @pytest.mark.fast
    def test_get_excluded_sources_binary_search_right_branch(self):
        """Test binary search takes right branch (position >= window_start)."""
        tracker = TimelineVarietyTracker(timeline_window=100.0, max_repeats=1)

        # Add sources
        tracker.record_usage("V1", "/video/a.mp4", 50.0)
        tracker.record_usage("V1", "/video/b.mp4", 60.0)

        # At position 100, window is [0, 100)
        # Both sources should be in window
        excluded = tracker.get_excluded_sources("V1", 100.0)

        assert "/video/a.mp4" in excluded
        assert "/video/b.mp4" in excluded

    @pytest.mark.fast
    def test_get_excluded_sources_break_at_current_position(self):
        """Test that loop breaks when position >= current_timeline_pos (line 80)."""
        tracker = TimelineVarietyTracker(timeline_window=100.0, max_repeats=1)

        # Add sources before and after current position
        tracker.record_usage("V1", "/video/a.mp4", 50.0)
        tracker.record_usage("V1", "/video/b.mp4", 100.0)  # At current position
        tracker.record_usage("V1", "/video/c.mp4", 150.0)  # After current position

        # At position 100, only source at 50 should be counted
        excluded = tracker.get_excluded_sources("V1", 100.0)

        assert "/video/a.mp4" in excluded
        # Source at 100 and 150 should not be counted (break)
        # But exclusion is based on count reaching max_repeats
        # b and c should not contribute to exclusion

    @pytest.mark.fast
    def test_get_excluded_sources_multiple_tracks(self):
        """Test that different tracks are independent."""
        tracker = TimelineVarietyTracker(timeline_window=100.0, max_repeats=1)

        tracker.record_usage("V1", "/video/a.mp4", 50.0)
        tracker.record_usage("V2", "/video/a.mp4", 50.0)

        # Track V1 should exclude a.mp4
        excluded_v1 = tracker.get_excluded_sources("V1", 100.0)
        assert "/video/a.mp4" in excluded_v1

        # Track V3 has no usage, nothing excluded
        excluded_v3 = tracker.get_excluded_sources("V3", 100.0)
        assert len(excluded_v3) == 0

    @pytest.mark.fast
    def test_get_stats_empty(self):
        """Test get_stats with no usage."""
        tracker = TimelineVarietyTracker()
        stats = tracker.get_stats()
        assert stats == {}

    @pytest.mark.fast
    def test_get_stats_with_usage(self):
        """Test get_stats with recorded usage."""
        tracker = TimelineVarietyTracker()

        tracker.record_usage("V1", "/path/video_a.mp4", 0.0)
        tracker.record_usage("V1", "/path/video_a.mp4", 10.0)
        tracker.record_usage("V1", "/path/video_b.mp4", 20.0)
        tracker.record_usage("V2", "/path/video_c.mp4", 0.0)

        stats = tracker.get_stats()

        assert "V1" in stats
        assert stats["V1"]["total_clips"] == 3
        assert stats["V1"]["unique_sources"] == 2
        assert len(stats["V1"]["top_sources"]) == 2

        assert "V2" in stats
        assert stats["V2"]["total_clips"] == 1
        assert stats["V2"]["unique_sources"] == 1


class TestGlobalClipTracker:
    """Test GlobalClipTracker class."""

    @pytest.mark.fast
    def test_init(self):
        """Test initialization."""
        tracker = GlobalClipTracker()
        assert len(tracker.used_clips) == 0
        assert len(tracker.clip_track_map) == 0

    @pytest.mark.fast
    def test_get_clip_id_regular_file(self):
        """Test clip ID generation for regular video file."""
        tracker = GlobalClipTracker()

        segment = MagicMock(spec=SRTSegment)
        segment.source_file = "/path/to/video.mp4"
        segment.start_time = 10.5
        segment.end_time = 25.3

        clip_id = tracker.get_clip_id(segment)

        # Should include path and time range
        assert "/path/to/video.mp4:10.50-25.30" in clip_id

    @pytest.mark.fast
    def test_get_clip_id_windows_path(self):
        """Test clip ID normalizes Windows paths."""
        tracker = GlobalClipTracker()

        segment = MagicMock(spec=SRTSegment)
        segment.source_file = "C:\\Users\\video.mp4"
        segment.start_time = 0.0
        segment.end_time = 10.0

        clip_id = tracker.get_clip_id(segment)

        # Backslashes should be converted to forward slashes
        assert "\\" not in clip_id
        assert "/" in clip_id

    @pytest.mark.fast
    def test_get_clip_id_audio_first_pattern(self):
        """Test clip ID for audio-first segment files (lines 163-167)."""
        tracker = GlobalClipTracker()

        # Audio-first pattern: {video_id}_{offset:04d}.mp4
        # e.g., abc12345678_0045.mp4 means 45 second offset
        segment = MagicMock(spec=SRTSegment)
        segment.source_file = "/videos/dQw4w9WgXcQ_0045.mp4"  # 11-char video ID, 4-digit offset
        segment.start_time = 5.0  # Start within segment
        segment.end_time = 15.0

        clip_id = tracker.get_clip_id(segment)

        # Original coordinates: offset (45) + segment times
        # original_start = 45 + 5 = 50.0
        # original_end = 45 + 15 = 60.0
        assert "dQw4w9WgXcQ:50.00-60.00" == clip_id

    @pytest.mark.fast
    def test_get_clip_id_audio_first_pattern_different_offsets(self):
        """Test different audio-first segment offsets."""
        tracker = GlobalClipTracker()

        # Test offset of 0 - must be exactly 11-char video ID + 4-digit offset
        segment = MagicMock(spec=SRTSegment)
        segment.source_file = "/videos/abc12345678_0000.mp4"
        segment.start_time = 0.0
        segment.end_time = 30.0

        clip_id = tracker.get_clip_id(segment)
        assert "abc12345678:0.00-30.00" == clip_id

        # Test larger offset with valid 11-char ID (like YouTube)
        segment.source_file = "/videos/xYz_abc-123_1234.mp4"  # exactly 11 chars: x-Y-z-_-a-b-c---1-2-3
        segment.start_time = 10.0
        segment.end_time = 20.0

        clip_id = tracker.get_clip_id(segment)
        # offset = 1234, original_start = 1234 + 10 = 1244
        assert "xYz_abc-123:1244.00-1254.00" == clip_id

    @pytest.mark.fast
    def test_get_clip_id_non_matching_pattern(self):
        """Test clip ID when filename doesn't match audio-first pattern."""
        tracker = GlobalClipTracker()

        # Filename that looks similar but doesn't match pattern
        segment = MagicMock(spec=SRTSegment)
        segment.source_file = "/videos/short_123.mp4"  # ID too short
        segment.start_time = 0.0
        segment.end_time = 10.0

        clip_id = tracker.get_clip_id(segment)

        # Should fall back to regular format
        assert "short_123" in clip_id or "/videos/short_123" in clip_id

    @pytest.mark.fast
    def test_is_used_false(self):
        """Test is_used returns False for unused clip."""
        tracker = GlobalClipTracker()

        segment = MagicMock(spec=SRTSegment)
        segment.source_file = "/video/test.mp4"
        segment.start_time = 0.0
        segment.end_time = 10.0

        assert tracker.is_used(segment) is False

    @pytest.mark.fast
    def test_is_used_true(self):
        """Test is_used returns True after recording usage."""
        tracker = GlobalClipTracker()

        segment = MagicMock(spec=SRTSegment)
        segment.source_file = "/video/test.mp4"
        segment.start_time = 0.0
        segment.end_time = 10.0

        tracker.record_usage(segment, "V1", 0)

        assert tracker.is_used(segment) is True

    @pytest.mark.fast
    def test_record_usage(self):
        """Test recording clip usage."""
        tracker = GlobalClipTracker()

        segment = MagicMock(spec=SRTSegment)
        segment.source_file = "/video/test.mp4"
        segment.start_time = 0.0
        segment.end_time = 10.0

        tracker.record_usage(segment, "V1", 5)

        assert len(tracker.used_clips) == 1
        clip_id = tracker.get_clip_id(segment)
        assert clip_id in tracker.used_clips
        assert tracker.clip_track_map[clip_id] == "V1@S005"

    @pytest.mark.fast
    def test_record_usage_multiple_clips(self):
        """Test recording multiple clips."""
        tracker = GlobalClipTracker()

        for i in range(3):
            segment = MagicMock(spec=SRTSegment)
            segment.source_file = f"/video/test_{i}.mp4"
            segment.start_time = float(i * 10)
            segment.end_time = float((i + 1) * 10)
            tracker.record_usage(segment, f"V{i+1}", i)

        assert len(tracker.used_clips) == 3

    @pytest.mark.fast
    def test_get_used_clips(self):
        """Test getting used clips (line 185)."""
        tracker = GlobalClipTracker()

        segment1 = MagicMock(spec=SRTSegment)
        segment1.source_file = "/video/a.mp4"
        segment1.start_time = 0.0
        segment1.end_time = 10.0

        segment2 = MagicMock(spec=SRTSegment)
        segment2.source_file = "/video/b.mp4"
        segment2.start_time = 0.0
        segment2.end_time = 10.0

        tracker.record_usage(segment1, "V1", 0)
        tracker.record_usage(segment2, "V2", 1)

        used_clips = tracker.get_used_clips()

        assert len(used_clips) == 2
        # Should be a copy
        assert used_clips is not tracker.used_clips
        # Modifying the copy shouldn't affect original
        used_clips.add("fake_clip")
        assert len(tracker.used_clips) == 2

    @pytest.mark.fast
    def test_get_stats(self):
        """Test getting tracker statistics."""
        tracker = GlobalClipTracker()

        segment1 = MagicMock(spec=SRTSegment)
        segment1.source_file = "/video/a.mp4"
        segment1.start_time = 0.0
        segment1.end_time = 10.0

        segment2 = MagicMock(spec=SRTSegment)
        segment2.source_file = "/video/b.mp4"
        segment2.start_time = 0.0
        segment2.end_time = 10.0

        tracker.record_usage(segment1, "V1", 0)
        tracker.record_usage(segment2, "V1", 1)

        stats = tracker.get_stats()

        assert stats["total_clips_used"] == 2
        assert stats["tracks_used"] == 1  # Both on V1

    @pytest.mark.fast
    def test_get_stats_multiple_tracks(self):
        """Test stats with multiple tracks."""
        tracker = GlobalClipTracker()

        for i in range(5):
            segment = MagicMock(spec=SRTSegment)
            segment.source_file = f"/video/{i}.mp4"
            segment.start_time = 0.0
            segment.end_time = 10.0
            track = f"V{(i % 3) + 1}"  # V1, V2, V3, V1, V2
            tracker.record_usage(segment, track, i)

        stats = tracker.get_stats()

        assert stats["total_clips_used"] == 5
        assert stats["tracks_used"] == 3  # V1, V2, V3


class TestTrackingEdgeCases:
    """Test edge cases for tracking classes."""

    @pytest.mark.fast
    def test_variety_tracker_same_position(self):
        """Test handling multiple clips at same position."""
        tracker = TimelineVarietyTracker(timeline_window=100.0, max_repeats=2)

        # Add multiple sources at same position
        tracker.record_usage("V1", "/video/a.mp4", 50.0)
        tracker.record_usage("V1", "/video/b.mp4", 50.0)

        excluded = tracker.get_excluded_sources("V1", 100.0)

        # Neither should be excluded since each only appears once
        assert "/video/a.mp4" not in excluded
        assert "/video/b.mp4" not in excluded

    @pytest.mark.fast
    def test_variety_tracker_max_repeats_boundary(self):
        """Test exclusion at exact max_repeats boundary."""
        tracker = TimelineVarietyTracker(timeline_window=100.0, max_repeats=3)

        # Add source exactly 3 times
        tracker.record_usage("V1", "/video/a.mp4", 10.0)
        tracker.record_usage("V1", "/video/a.mp4", 20.0)
        tracker.record_usage("V1", "/video/a.mp4", 30.0)

        excluded = tracker.get_excluded_sources("V1", 50.0)

        # Should be excluded (count == max_repeats)
        assert "/video/a.mp4" in excluded

    @pytest.mark.fast
    def test_global_tracker_same_clip_different_tracks(self):
        """Test same clip used on different tracks."""
        tracker = GlobalClipTracker()

        segment = MagicMock(spec=SRTSegment)
        segment.source_file = "/video/test.mp4"
        segment.start_time = 0.0
        segment.end_time = 10.0

        # Record on first track
        tracker.record_usage(segment, "V1", 0)

        # Same clip should be marked as used
        assert tracker.is_used(segment) is True

        # Recording again should still work but not duplicate
        tracker.record_usage(segment, "V2", 1)

        # Set only counts unique clips
        assert len(tracker.used_clips) == 1

    @pytest.mark.fast
    def test_audio_first_pattern_edge_cases(self):
        """Test edge cases for audio-first pattern matching."""
        tracker = GlobalClipTracker()

        # Test with all letters in video ID
        segment = MagicMock(spec=SRTSegment)
        segment.source_file = "/path/abcdefghijk_0100.mp4"  # 11 letters
        segment.start_time = 5.0
        segment.end_time = 15.0

        clip_id = tracker.get_clip_id(segment)
        # offset = 100, original_start = 100 + 5 = 105
        assert "abcdefghijk:105.00-115.00" == clip_id

    @pytest.mark.fast
    def test_clip_id_precision(self):
        """Test clip ID uses 2 decimal precision."""
        tracker = GlobalClipTracker()

        segment = MagicMock(spec=SRTSegment)
        segment.source_file = "/video/test.mp4"
        segment.start_time = 1.234567
        segment.end_time = 2.999999

        clip_id = tracker.get_clip_id(segment)

        # Should be rounded to 2 decimals
        assert "1.23-3.00" in clip_id
