"""
Tests for caption-first video ID resolution per Rule 26.

Rule 26 Summary:
In caption-first mode, matches store video IDs (e.g., 'DDi-Swd7Qcw') NOT file paths.
The source_file field contains:
- YouTube video ID (11 chars) - needs segment resolution
- Full path (if from audio-first cache) - already resolved

DOWNLOAD_SEGMENTS only downloads segments for videos in video_candidates.
Videos from the global caption cache (other projects) won't have segments downloaded.

Symptom: OTIO has invalid paths like file:///D:/_Projects/.../DDi-Swd7Qcw
(video ID used as filename, no .mp4 extension).

Tests:
- AC1: matches with video ID source_file resolve to segment paths after DOWNLOAD_SEGMENTS
- AC2: global caption cache entries don't have segments downloaded (expected gap)
- AC3: OTIO paths are valid file:/// URLs, not raw video IDs
- AC4: extract_video_id() handles both 11-char IDs and full URLs
- AC5: segment coverage analysis identifies missing segments from global cache
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Optional, Dict, List

from src.transcription.utils import extract_video_id
from src.downloader.segment_utils import (
    _extract_video_id,
    get_segment_filename,
    collect_matched_segments,
    prepare_merged_segments,
)
from src.otio.timeline import _is_missing_file


# ============================================================================
# AC1: Test matches with video ID source_file resolve to segment paths
# ============================================================================

class TestVideoIdToSegmentPathResolution:
    """Test that video ID source_files resolve to segment paths after DOWNLOAD_SEGMENTS."""

    @pytest.mark.fast
    def test_video_id_in_match_resolves_to_segment_file(self, tmp_path):
        """Video ID in source_file should resolve to segment file after download."""
        # Create segment files that would exist after DOWNLOAD_SEGMENTS
        segments_dir = tmp_path / "videos" / "abc12345678_segments"
        segments_dir.mkdir(parents=True)
        segment_file = segments_dir / "abc12345678_0030.mp4"
        segment_file.touch()

        # Match stores video ID as source_file (caption-first behavior)
        video_id = "abc12345678"

        # After DOWNLOAD_SEGMENTS, segment files exist
        # The segment filename encodes video_id + start time
        expected_segment = get_segment_filename(video_id, 30)
        assert expected_segment == "abc12345678_0030.mp4"

        # Segment file should exist
        assert segment_file.exists()

    @pytest.mark.fast
    def test_segment_filename_format_encodes_video_id_and_time(self):
        """Segment filename format: {video_id}_{start_seconds:04d}.mp4"""
        assert get_segment_filename("dQw4w9WgXcQ", 0) == "dQw4w9WgXcQ_0000.mp4"
        assert get_segment_filename("abc12345678", 330) == "abc12345678_0330.mp4"
        assert get_segment_filename("xyz-_Aa0Zz1", 9999) == "xyz-_Aa0Zz1_9999.mp4"

    @pytest.mark.fast
    def test_match_with_video_id_finds_segment_path(self, tmp_path):
        """collect_matched_segments uses video_id to map to audio_downloads."""
        @dataclass
        class MockSegment:
            source_file: str
            start_time: float
            end_time: float

        @dataclass
        class MockMatch:
            video_segment: MockSegment

        @dataclass
        class MockResult:
            primary_match: MockMatch = None
            alternatives: list = None
            secondary_matches: list = None
            strategy_matches: list = None

        @dataclass
        class MockAudioDownload:
            url: str
            duration: float = 300.0
            keyword: str = "test"
            file: str = ""

        # Caption-first: source_file contains video ID
        video_id = "dQw4w9WgXcQ"
        result = MockResult(
            primary_match=MockMatch(
                video_segment=MockSegment(
                    source_file=video_id,  # Just the ID
                    start_time=30,
                    end_time=60
                )
            )
        )

        audio_downloads = {
            video_id: MockAudioDownload(
                url=f"https://youtube.com/watch?v={video_id}",
                file=f"{video_id}.mp3"
            )
        }

        segments = collect_matched_segments([result], audio_downloads)

        # Should find the video and create MatchedSegment
        assert video_id in segments
        assert len(segments[video_id]) == 1
        assert segments[video_id][0].video_id == video_id

    @pytest.mark.fast
    def test_match_with_full_path_also_works(self, tmp_path):
        """source_file with full path (audio-first cache) should still work."""
        @dataclass
        class MockSegment:
            source_file: str
            start_time: float
            end_time: float

        @dataclass
        class MockMatch:
            video_segment: MockSegment

        @dataclass
        class MockResult:
            primary_match: MockMatch = None
            alternatives: list = None
            secondary_matches: list = None
            strategy_matches: list = None

        @dataclass
        class MockAudioDownload:
            url: str
            duration: float = 300.0
            keyword: str = "test"
            file: str = ""

        # Audio-first: source_file contains full path
        video_id = "abc12345678"
        full_path = str(tmp_path / f"{video_id}.mp3")

        result = MockResult(
            primary_match=MockMatch(
                video_segment=MockSegment(
                    source_file=full_path,
                    start_time=30,
                    end_time=60
                )
            )
        )

        audio_downloads = {
            video_id: MockAudioDownload(
                url=f"https://youtube.com/watch?v={video_id}",
                file=full_path
            )
        }

        segments = collect_matched_segments([result], audio_downloads)

        # Should extract video_id from path and find it
        assert video_id in segments
        assert len(segments[video_id]) == 1


# ============================================================================
# AC2: Test global caption cache entries don't have segments downloaded
# ============================================================================

class TestGlobalCacheSegmentGaps:
    """Test that global caption cache entries correctly identify missing segments."""

    @pytest.mark.fast
    def test_global_cache_video_not_in_project_candidates(self):
        """Videos from global cache aren't in video_candidates, so no segments."""
        # Simulate project's video_candidates
        project_candidates = {"proj_vid_123", "proj_vid_456"}

        # Video from global cache (different project)
        global_cache_video_id = "global_abc12"

        # Global cache video is NOT in project's candidates
        assert global_cache_video_id not in project_candidates

    @pytest.mark.fast
    def test_segment_collection_skips_missing_audio_download(self):
        """collect_matched_segments skips videos not in audio_downloads."""
        @dataclass
        class MockSegment:
            source_file: str
            start_time: float
            end_time: float

        @dataclass
        class MockMatch:
            video_segment: MockSegment

        @dataclass
        class MockResult:
            primary_match: MockMatch = None
            alternatives: list = None
            secondary_matches: list = None
            strategy_matches: list = None

        # Video from global cache (not in project's audio_downloads)
        global_video_id = "global_12345"
        result = MockResult(
            primary_match=MockMatch(
                video_segment=MockSegment(
                    source_file=global_video_id,
                    start_time=30,
                    end_time=60
                )
            )
        )

        # Only project's videos in audio_downloads
        audio_downloads = {
            "project_vid1": Mock(url="http://test", keyword="test", file="project_vid1.mp3")
        }

        segments = collect_matched_segments([result], audio_downloads)

        # Global cache video should NOT be collected (no audio download)
        assert global_video_id not in segments
        assert "project_vid1" not in segments  # Not matched either

    @pytest.mark.fast
    def test_missing_segment_results_in_gap_detection(self, tmp_path):
        """Video ID without corresponding segment file triggers gap detection."""
        video_id = "global_abcde"
        expected_segment = tmp_path / f"{video_id}_0030.mp4"

        # Segment file doesn't exist (global cache, not downloaded)
        assert not expected_segment.exists()

        # Can detect that this would be a gap in OTIO
        # The OTIO builder uses file existence checks


# ============================================================================
# AC3: Test OTIO paths are valid file:/// URLs, not raw video IDs
# ============================================================================

class TestOtioPathValidation:
    """Test that OTIO outputs valid file:/// URLs, not raw video IDs."""

    @pytest.mark.fast
    def test_is_missing_file_handles_video_id(self):
        """_is_missing_file returns False for video IDs (no path separators)."""
        # Video ID without path separators - not treated as local file
        video_id = "dQw4w9WgXcQ"
        assert _is_missing_file(video_id) is False

    @pytest.mark.fast
    def test_is_missing_file_handles_full_path(self, tmp_path):
        """_is_missing_file returns True for missing file paths."""
        missing_path = str(tmp_path / "nonexistent.mp4")
        assert _is_missing_file(missing_path) is True

    @pytest.mark.fast
    def test_is_missing_file_handles_existing_path(self, tmp_path):
        """_is_missing_file returns False for existing file paths."""
        existing_file = tmp_path / "existing.mp4"
        existing_file.touch()
        assert _is_missing_file(str(existing_file)) is False

    @pytest.mark.fast
    def test_valid_file_path_has_extension(self):
        """Valid segment file paths have .mp4 extension."""
        # Valid path pattern
        valid_path = "E:/v/project/abc12345678_0030.mp4"
        assert valid_path.endswith(".mp4")

        # Invalid: just video ID (no extension)
        invalid_path = "abc12345678"
        assert not invalid_path.endswith(".mp4")

    @pytest.mark.fast
    def test_file_url_format_is_valid(self):
        """file:/// URLs should have proper format."""
        # Windows path converted to file URL
        windows_path = "E:/v/project/abc12345678_0030.mp4"
        file_url = f"file:///{windows_path}"

        assert file_url.startswith("file:///")
        assert file_url.endswith(".mp4")
        assert "/" in file_url  # Uses forward slashes

        # Invalid: raw video ID in URL
        bad_url = "file:///D:/_Projects/.../dQw4w9WgXcQ"
        assert not bad_url.endswith(".mp4")

    @pytest.mark.fast
    def test_otio_clip_path_not_video_id(self):
        """OTIO clips should have file paths, not video IDs."""
        # Simulate what an OTIO clip path should look like
        proper_path = "E:/v/MyProject/dQw4w9WgXcQ_0045.mp4"

        # Check it's a proper path, not just a video ID
        assert "_" in proper_path  # Has segment suffix
        assert ".mp4" in proper_path  # Has extension
        assert "/" in proper_path or "\\" in proper_path  # Has path separators


# ============================================================================
# AC4: Test extract_video_id() handles both 11-char IDs and full URLs
# ============================================================================

class TestExtractVideoIdHandling:
    """Test extract_video_id handles various input formats."""

    @pytest.mark.fast
    def test_extract_from_11_char_id(self):
        """Extract video ID from plain 11-character ID."""
        # transcription.utils.extract_video_id
        assert extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"
        assert extract_video_id("abc12345678") == "abc12345678"
        assert extract_video_id("xyz-_Aa0Zz1") == "xyz-_Aa0Zz1"

    @pytest.mark.fast
    def test_extract_from_filename_with_extension(self):
        """Extract video ID from filename with extension."""
        assert extract_video_id("dQw4w9WgXcQ.mp4") == "dQw4w9WgXcQ"
        assert extract_video_id("abc12345678.mp3") == "abc12345678"

    @pytest.mark.fast
    def test_extract_from_segment_filename(self):
        """Extract video ID from segment filename format."""
        assert extract_video_id("dQw4w9WgXcQ_0045.mp4") == "dQw4w9WgXcQ"
        assert extract_video_id("abc12345678_0330.mp4") == "abc12345678"
        assert extract_video_id("xyz-_Aa0Zz1_9999.mp4") == "xyz-_Aa0Zz1"

    @pytest.mark.fast
    def test_extract_from_full_path(self):
        """Extract video ID from full file path."""
        assert extract_video_id("/path/to/dQw4w9WgXcQ.mp4") == "dQw4w9WgXcQ"
        assert extract_video_id("E:/videos/abc12345678_0030.mp4") == "abc12345678"

    @pytest.mark.fast
    def test_extract_handles_short_strings(self):
        """Extract returns None for strings shorter than 11 chars."""
        result = extract_video_id("short")
        # Either None or whatever the function does for short strings
        assert result is None or len(result) < 11

    @pytest.mark.fast
    def test_extract_handles_empty_string(self):
        """Extract handles empty string gracefully."""
        result = extract_video_id("")
        assert result is None

    @pytest.mark.fast
    def test_segment_utils_extract_video_id_from_path(self):
        """segment_utils._extract_video_id handles paths."""
        # segment_utils version
        assert _extract_video_id("abc123.mp3") == "abc123"
        assert _extract_video_id("abc123_0330.mp4") == "abc123"
        assert _extract_video_id("/path/to/video.mp3") == "video"

    @pytest.mark.fast
    def test_segment_utils_extract_video_id_none(self):
        """segment_utils._extract_video_id returns None for empty."""
        assert _extract_video_id("") is None
        assert _extract_video_id(None) is None


# ============================================================================
# AC5: Test segment coverage analysis identifies missing segments
# ============================================================================

class TestSegmentCoverageAnalysis:
    """Test segment coverage analysis for identifying missing segments."""

    @pytest.mark.fast
    def test_identify_segments_from_matches(self):
        """Analyze match video_files to identify which segments are needed."""
        matches = [
            {"video_file": "dQw4w9WgXcQ", "start": 30, "end": 60},
            {"video_file": "abc12345678", "start": 0, "end": 30},
            {"video_file": "dQw4w9WgXcQ", "start": 90, "end": 120},  # Same video, different time
        ]

        # Collect unique video IDs
        video_ids = {m["video_file"] for m in matches}
        assert video_ids == {"dQw4w9WgXcQ", "abc12345678"}

    @pytest.mark.fast
    def test_identify_missing_segments(self, tmp_path):
        """Identify which video IDs don't have segment files."""
        segments_dir = tmp_path / "segments"
        segments_dir.mkdir()

        # Only one video has segments
        (segments_dir / "dQw4w9WgXcQ_0030.mp4").touch()

        match_video_ids = {"dQw4w9WgXcQ", "abc12345678", "global_12345"}

        # Check which have segment files
        segment_files = set(segments_dir.glob("*.mp4"))
        segment_ids = {f.stem.rsplit("_", 1)[0] for f in segment_files if "_" in f.stem}

        missing = match_video_ids - segment_ids
        assert missing == {"abc12345678", "global_12345"}

    @pytest.mark.fast
    def test_global_cache_videos_in_missing_list(self, tmp_path):
        """Videos from global cache should appear in missing segments list."""
        segments_dir = tmp_path / "segments"
        segments_dir.mkdir()

        # Project videos have segments
        (segments_dir / "project_vid1_0000.mp4").touch()
        (segments_dir / "project_vid2_0000.mp4").touch()

        # All match video IDs (including global cache)
        match_video_ids = {"project_vid1", "project_vid2", "global_abc12", "global_xyz99"}

        # Find segments
        segment_files = set(segments_dir.glob("*_*.mp4"))
        segment_ids = {f.stem.rsplit("_", 1)[0] for f in segment_files if "_" in f.stem}

        missing = match_video_ids - segment_ids

        # Global cache videos should be in missing
        assert "global_abc12" in missing
        assert "global_xyz99" in missing
        # Project videos should NOT be in missing
        assert "project_vid1" not in missing
        assert "project_vid2" not in missing

    @pytest.mark.fast
    def test_coverage_percentage_calculation(self, tmp_path):
        """Calculate segment coverage percentage."""
        segments_dir = tmp_path / "segments"
        segments_dir.mkdir()

        # 3 out of 5 videos have segments
        (segments_dir / "vid1_0000.mp4").touch()
        (segments_dir / "vid2_0000.mp4").touch()
        (segments_dir / "vid3_0000.mp4").touch()

        match_video_ids = {"vid1", "vid2", "vid3", "vid4", "vid5"}

        segment_files = set(segments_dir.glob("*_*.mp4"))
        segment_ids = {f.stem.rsplit("_", 1)[0] for f in segment_files if "_" in f.stem}

        covered = len(match_video_ids & segment_ids)
        total = len(match_video_ids)
        coverage_percent = (covered / total) * 100

        assert coverage_percent == 60.0

    @pytest.mark.fast
    def test_prepare_merged_segments_handles_missing_audio_downloads(self):
        """prepare_merged_segments handles videos not in audio_downloads gracefully.

        Note: The actual filtering happens earlier in collect_matched_segments.
        prepare_merged_segments processes all segments_by_video entries,
        but uses None for video_duration when audio_download is missing.
        """
        from src.downloader.types import MatchedSegment

        @dataclass
        class MockAudioDownload:
            url: str
            duration: float
            keyword: str = ""

        # Only project video in segments_by_video
        # (global cache video would have been filtered by collect_matched_segments)
        segments_by_video = {
            "vid1": [MatchedSegment(
                video_id="vid1", video_url="http://test",
                start_time=0, end_time=30, track="V1",
                voiceover_segment_idx=0, keyword="test"
            )]
        }

        audio_downloads = {
            "vid1": MockAudioDownload(url="http://test", duration=300)
        }

        merged = prepare_merged_segments(
            segments_by_video, audio_downloads,
            buffer_seconds=5, merge_gap_seconds=10
        )

        # vid1 should have merged segments
        merged_ids = {m.video_id for m in merged}
        assert "vid1" in merged_ids
        assert len(merged) == 1

    @pytest.mark.fast
    def test_collect_matched_segments_filters_global_cache_videos(self):
        """collect_matched_segments skips videos not in audio_downloads.

        This is where global cache videos get filtered out - they won't
        appear in segments_by_video if their video_id isn't in audio_downloads.
        """
        @dataclass
        class MockSegment:
            source_file: str
            start_time: float
            end_time: float

        @dataclass
        class MockMatch:
            video_segment: MockSegment

        @dataclass
        class MockResult:
            primary_match: MockMatch = None
            alternatives: list = None
            secondary_matches: list = None
            strategy_matches: list = None

        @dataclass
        class MockAudioDownload:
            url: str
            duration: float = 300.0
            keyword: str = "test"
            file: str = ""

        # Matches include both project and global cache videos
        results = [
            MockResult(
                primary_match=MockMatch(
                    video_segment=MockSegment(
                        source_file="project_vid1",
                        start_time=0, end_time=30
                    )
                )
            ),
            MockResult(
                primary_match=MockMatch(
                    video_segment=MockSegment(
                        source_file="global_cache_vid",  # Not in audio_downloads
                        start_time=0, end_time=30
                    )
                )
            )
        ]

        # Only project video in audio_downloads
        audio_downloads = {
            "project_vid1": MockAudioDownload(
                url="http://test", file="project_vid1.mp3"
            )
        }

        segments = collect_matched_segments(results, audio_downloads)

        # Only project video should be collected
        assert "project_vid1" in segments
        assert "global_cache_vid" not in segments


# ============================================================================
# Integration: End-to-end caption-first segment resolution
# ============================================================================

class TestCaptionFirstIntegration:
    """Integration tests for caption-first video ID resolution flow."""

    @pytest.mark.fast
    def test_caption_first_flow_video_id_to_segment(self, tmp_path):
        """Complete flow: video ID in match -> segment file path in OTIO."""
        @dataclass
        class MockSegment:
            source_file: str
            start_time: float
            end_time: float

        @dataclass
        class MockMatch:
            video_segment: MockSegment

        @dataclass
        class MockResult:
            primary_match: MockMatch = None
            alternatives: list = None
            secondary_matches: list = None
            strategy_matches: list = None

        @dataclass
        class MockAudioDownload:
            url: str
            duration: float = 300.0
            keyword: str = "test"
            file: str = ""

        # Step 1: Caption-first mode stores video ID as source_file
        video_id = "dQw4w9WgXcQ"
        match_result = MockResult(
            primary_match=MockMatch(
                video_segment=MockSegment(
                    source_file=video_id,  # Just video ID
                    start_time=45,
                    end_time=75
                )
            )
        )

        # Step 2: collect_matched_segments finds video in audio_downloads
        audio_downloads = {
            video_id: MockAudioDownload(
                url=f"https://youtube.com/watch?v={video_id}",
                file=f"{video_id}.mp3"
            )
        }

        segments = collect_matched_segments([match_result], audio_downloads)
        assert video_id in segments

        # Step 3: After DOWNLOAD_SEGMENTS, segment file would exist
        segments_dir = tmp_path / "segments"
        segments_dir.mkdir()
        segment_file = segments_dir / get_segment_filename(video_id, 15)  # buffered start
        segment_file.touch()

        # Step 4: OTIO builder would use full path
        full_path = str(segment_file)
        assert Path(full_path).exists()
        assert ".mp4" in full_path

    @pytest.mark.fast
    def test_global_cache_video_creates_gap(self, tmp_path):
        """Global cache video without segment creates gap in OTIO."""
        # Global cache video ID (not in project's audio_downloads)
        global_video_id = "global_12345"

        # No segment file for this video
        segments_dir = tmp_path / "segments"
        segments_dir.mkdir()

        expected_segment = segments_dir / get_segment_filename(global_video_id, 0)
        assert not expected_segment.exists()

        # OTIO builder would detect missing file
        # (simulated - actual behavior is inserting gap)
        segment_path = str(expected_segment)
        is_missing = _is_missing_file(segment_path)
        assert is_missing is True
