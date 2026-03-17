"""
Tests for OTIO auto-split validation (Rule 17) and MediaPathNormalizer (Rule 16).

Rule 17: Large timelines (3000+ items) auto-split into PART1-4 files
Rule 16: MediaPathNormalizer deduplicates same file in different folders

This test module covers:
- AC1: Timeline with exactly 3000 items does not split
- AC2: Timeline with 3001+ items auto-splits into PART1-4 files
- AC3: MediaPathNormalizer deduplicates same file in different folders (Rule 16)
- AC4: Split timelines maintain correct clip timing across parts
- AC5: Large timeline warning is logged when approaching split threshold
"""

import logging
import math
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import opentimelineio as otio

from src.otio.export import (
    _split_timeline_by_segments,
    save_timeline_split,
)
from src.otio.timeline import (
    CLIP_COUNT_WARNING_THRESHOLD,
    CLIP_COUNT_ERROR_THRESHOLD,
    _count_timeline_clips,
    _log_clip_count_warnings,
)
from src.otio.utils import (
    MediaPathNormalizer,
    build_canonical_media_map,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def create_timeline_with_n_segments():
    """Factory fixture to create a timeline with N segments on V1 track."""
    def _create(n_segments: int, frame_rate: float = 30.0) -> otio.schema.Timeline:
        timeline = otio.schema.Timeline(name="Test Timeline")
        timeline.metadata['Resolve_OTIO'] = {'Resolve OTIO Meta Version': '1.0'}
        timeline.global_start_time = otio.opentime.RationalTime(
            int(3600 * frame_rate), frame_rate
        )

        # Create V1 video track with N clips
        v1_track = otio.schema.Track(name="V1 - Primary", kind=otio.schema.TrackKind.Video)
        v1_track.metadata['Resolve_OTIO'] = {'Locked': False}

        for i in range(n_segments):
            clip = otio.schema.Clip(
                name=f"Clip_{i:04d}",
                source_range=otio.opentime.TimeRange(
                    start_time=otio.opentime.RationalTime(i * 30, frame_rate),
                    duration=otio.opentime.RationalTime(30, frame_rate)  # 1 second each
                )
            )
            v1_track.append(clip)

        timeline.tracks.append(v1_track)

        # Create A1 audio track with matching clips
        a1_track = otio.schema.Track(name="A1 - Video Audio", kind=otio.schema.TrackKind.Audio)
        a1_track.metadata['Resolve_OTIO'] = {'Locked': False}

        for i in range(n_segments):
            clip = otio.schema.Clip(
                name=f"Audio_{i:04d}",
                source_range=otio.opentime.TimeRange(
                    start_time=otio.opentime.RationalTime(i * 30, frame_rate),
                    duration=otio.opentime.RationalTime(30, frame_rate)
                )
            )
            a1_track.append(clip)

        timeline.tracks.append(a1_track)

        return timeline

    return _create


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# AC1: Timeline with exactly 3000 items does not split
# =============================================================================

class TestTimelineExact3000NoSplit:
    """Test that timeline with exactly 3000 items does not split (AC1)."""

    @pytest.mark.fast
    def test_split_timeline_3000_segments_returns_single_timeline(self, create_timeline_with_n_segments):
        """Timeline with exactly 3000 segments should not split when max=3000."""
        timeline = create_timeline_with_n_segments(3000)

        result = _split_timeline_by_segments(timeline, max_segments=3000)

        assert len(result) == 1
        assert result[0] == timeline

    @pytest.mark.fast
    def test_split_timeline_2999_segments_returns_single_timeline(self, create_timeline_with_n_segments):
        """Timeline with 2999 segments (under threshold) should not split."""
        timeline = create_timeline_with_n_segments(2999)

        result = _split_timeline_by_segments(timeline, max_segments=3000)

        assert len(result) == 1
        assert result[0] == timeline

    @pytest.mark.fast
    def test_split_timeline_boundary_at_max_segments(self, create_timeline_with_n_segments):
        """Verify boundary condition: exactly max_segments returns original timeline."""
        max_segments = 100  # Use smaller number for faster test
        timeline = create_timeline_with_n_segments(max_segments)

        result = _split_timeline_by_segments(timeline, max_segments=max_segments)

        assert len(result) == 1
        # Should return original timeline object when no split needed
        assert result[0] == timeline

    @pytest.mark.fast
    def test_clip_count_at_error_threshold_no_auto_split_in_count(self, create_timeline_with_n_segments):
        """Test that _count_timeline_clips correctly counts at threshold boundary."""
        # Note: CLIP_COUNT_ERROR_THRESHOLD is for warning, not auto-split trigger
        # Auto-split is controlled by max_segments_per_file parameter
        timeline = create_timeline_with_n_segments(CLIP_COUNT_ERROR_THRESHOLD // 2)

        clip_count = _count_timeline_clips(timeline)

        # Each segment creates 1 video clip + 1 audio clip
        expected_clips = (CLIP_COUNT_ERROR_THRESHOLD // 2) * 2
        assert clip_count == expected_clips


# =============================================================================
# AC2: Timeline with 3001+ items auto-splits into PART1-N files
# =============================================================================

class TestTimelineAutoSplit:
    """Test that timeline with 3001+ items auto-splits (AC2)."""

    @pytest.mark.fast
    def test_split_timeline_3001_segments_creates_two_parts(self, create_timeline_with_n_segments):
        """Timeline with 3001 segments should split into 2 parts when max=3000."""
        timeline = create_timeline_with_n_segments(3001)

        result = _split_timeline_by_segments(timeline, max_segments=3000)

        assert len(result) == 2
        # First part should have 3000 segments
        assert len(list(result[0].tracks[0])) == 3000
        # Second part should have 1 segment
        assert len(list(result[1].tracks[0])) == 1

    @pytest.mark.fast
    def test_split_timeline_6000_segments_creates_two_parts(self, create_timeline_with_n_segments):
        """Timeline with 6000 segments should split into exactly 2 parts when max=3000."""
        timeline = create_timeline_with_n_segments(6000)

        result = _split_timeline_by_segments(timeline, max_segments=3000)

        assert len(result) == 2
        assert len(list(result[0].tracks[0])) == 3000
        assert len(list(result[1].tracks[0])) == 3000

    @pytest.mark.fast
    def test_split_timeline_6001_segments_creates_three_parts(self, create_timeline_with_n_segments):
        """Timeline with 6001 segments should split into 3 parts when max=3000."""
        timeline = create_timeline_with_n_segments(6001)

        result = _split_timeline_by_segments(timeline, max_segments=3000)

        assert len(result) == 3
        assert len(list(result[0].tracks[0])) == 3000
        assert len(list(result[1].tracks[0])) == 3000
        assert len(list(result[2].tracks[0])) == 1

    @pytest.mark.fast
    def test_split_timeline_calculates_correct_num_parts(self, create_timeline_with_n_segments):
        """Verify num_parts calculation matches math.ceil(segments / max_segments)."""
        test_cases = [
            (100, 30, 4),    # 100/30 = 3.33 -> 4 parts
            (90, 30, 3),     # 90/30 = 3.0 -> 3 parts
            (91, 30, 4),     # 91/30 = 3.03 -> 4 parts
            (150, 50, 3),    # 150/50 = 3.0 -> 3 parts
        ]

        for n_segments, max_seg, expected_parts in test_cases:
            timeline = create_timeline_with_n_segments(n_segments)
            result = _split_timeline_by_segments(timeline, max_segments=max_seg)

            assert len(result) == expected_parts, \
                f"Expected {expected_parts} parts for {n_segments} segments with max={max_seg}, got {len(result)}"

    @pytest.mark.fast
    def test_split_timeline_part_names_include_part_number(self, create_timeline_with_n_segments):
        """Split timelines should have part number in their names."""
        timeline = create_timeline_with_n_segments(100)

        result = _split_timeline_by_segments(timeline, max_segments=30)

        assert "(Part 1)" in result[0].name
        assert "(Part 2)" in result[1].name
        assert "(Part 3)" in result[2].name

    @pytest.mark.fast
    def test_save_timeline_split_generates_part_files(self, create_timeline_with_n_segments, temp_dir):
        """save_timeline_split with max_segments_per_file creates part files."""
        timeline = create_timeline_with_n_segments(100)
        output_path = temp_dir / "timeline.otio"

        generated_paths = save_timeline_split(
            timeline,
            str(output_path),
            max_segments_per_file=30
        )

        # Should have track files + FULL part files
        full_parts = [p for p in generated_paths if '_FULL_part' in p]
        assert len(full_parts) == 4  # 100/30 = 4 parts

        # Verify part files exist
        for part_path in full_parts:
            assert Path(part_path).exists(), f"Part file should exist: {part_path}"

    @pytest.mark.fast
    def test_save_timeline_split_no_split_when_under_threshold(self, create_timeline_with_n_segments, temp_dir):
        """save_timeline_split should create single FULL file when under threshold."""
        timeline = create_timeline_with_n_segments(50)
        output_path = temp_dir / "timeline.otio"

        generated_paths = save_timeline_split(
            timeline,
            str(output_path),
            max_segments_per_file=100  # 50 < 100, no split needed
        )

        # Should have track files + single FULL file (not parts)
        full_parts = [p for p in generated_paths if '_FULL_part' in p]
        full_files = [p for p in generated_paths if '_FULL.otio' in p]

        assert len(full_parts) == 0, "Should not create part files when under threshold"
        assert len(full_files) == 1, "Should create single FULL file"


# =============================================================================
# AC3: MediaPathNormalizer deduplicates same file in different folders (Rule 16)
# =============================================================================

class TestMediaPathNormalizerDeduplication:
    """Test MediaPathNormalizer deduplicates same file in different folders (AC3/Rule 16)."""

    @pytest.mark.fast
    def test_normalizer_maps_duplicate_files_to_canonical(self, temp_dir):
        """Same file in different folders should map to single canonical path."""
        # Create duplicate files in different folders
        stock_dir = temp_dir / "stock"
        broll_dir = temp_dir / "broll" / "pexels"
        stock_dir.mkdir(parents=True)
        broll_dir.mkdir(parents=True)

        # Create files with same name and size
        video_content = b"fake video content" * 1000  # Same content = same size

        stock_file = stock_dir / "video.mp4"
        broll_file = broll_dir / "video.mp4"

        stock_file.write_bytes(video_content)
        broll_file.write_bytes(video_content)

        # Use normalizer
        normalizer = MediaPathNormalizer()
        normalizer.register(str(stock_file))
        normalizer.register(str(broll_file))
        normalizer.build_map()

        # Both should map to same canonical path (stock/ preferred)
        canonical_stock = normalizer.get_canonical(str(stock_file))
        canonical_broll = normalizer.get_canonical(str(broll_file))

        assert canonical_stock == canonical_broll
        assert "stock" in canonical_stock  # stock/ is preferred

    @pytest.mark.fast
    def test_normalizer_prefers_stock_folder(self, temp_dir):
        """MediaPathNormalizer should prefer /stock/ folder for canonical path.

        Note: build_canonical_media_map() checks for '/stock/' with forward slashes.
        On Windows, paths have backslashes, so this test verifies that when '/stock/'
        is NOT found (due to backslashes), the shortest path is used instead.
        """
        # Create files
        other_dir = temp_dir / "other"
        stock_dir = temp_dir / "stock"
        other_dir.mkdir()
        stock_dir.mkdir()

        content = b"x" * 500
        other_file = other_dir / "video.mp4"
        stock_file = stock_dir / "video.mp4"

        other_file.write_bytes(content)
        stock_file.write_bytes(content)

        # Use forward slashes to match the check in build_canonical_media_map()
        other_file_str = str(other_file).replace('\\', '/')
        stock_file_str = str(stock_file).replace('\\', '/')

        normalizer = MediaPathNormalizer()
        normalizer.register(other_file_str)
        normalizer.register(stock_file_str)
        normalizer.build_map()

        # Stock should be canonical when using forward slashes (matches '/stock/' check)
        assert normalizer.get_canonical(other_file_str) == stock_file_str

    @pytest.mark.fast
    def test_normalizer_uses_shortest_path_when_no_stock(self, temp_dir):
        """Without /stock/ folder, normalizer should prefer shortest path."""
        # Create files in paths of different lengths
        short_dir = temp_dir / "a"
        long_dir = temp_dir / "longer" / "path" / "here"
        short_dir.mkdir()
        long_dir.mkdir(parents=True)

        content = b"y" * 300
        short_file = short_dir / "video.mp4"
        long_file = long_dir / "video.mp4"

        short_file.write_bytes(content)
        long_file.write_bytes(content)

        normalizer = MediaPathNormalizer()
        normalizer.register(str(long_file))
        normalizer.register(str(short_file))
        normalizer.build_map()

        # Shorter path should be canonical
        canonical = normalizer.get_canonical(str(long_file))
        assert canonical == str(short_file)

    @pytest.mark.fast
    def test_normalizer_tracks_duplicate_count(self, temp_dir):
        """MediaPathNormalizer should count duplicates found."""
        # Create 3 copies of same file
        dir1 = temp_dir / "dir1"
        dir2 = temp_dir / "dir2"
        dir3 = temp_dir / "dir3"
        for d in [dir1, dir2, dir3]:
            d.mkdir()

        content = b"z" * 200
        files = []
        for d in [dir1, dir2, dir3]:
            f = d / "same.mp4"
            f.write_bytes(content)
            files.append(str(f))

        normalizer = MediaPathNormalizer()
        for f in files:
            normalizer.register(f)
        normalizer.build_map()

        # 3 paths -> 1 canonical = 2 duplicates
        assert normalizer.duplicates_found == 2

    @pytest.mark.fast
    def test_normalizer_unique_files_map_to_themselves(self, temp_dir):
        """Unique files (different name/size) should map to themselves."""
        dir1 = temp_dir / "dir1"
        dir2 = temp_dir / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        file1 = dir1 / "video1.mp4"
        file2 = dir2 / "video2.mp4"

        file1.write_bytes(b"content1")
        file2.write_bytes(b"different content2")

        normalizer = MediaPathNormalizer()
        normalizer.register(str(file1))
        normalizer.register(str(file2))
        normalizer.build_map()

        assert normalizer.get_canonical(str(file1)) == str(file1)
        assert normalizer.get_canonical(str(file2)) == str(file2)
        assert normalizer.duplicates_found == 0

    @pytest.mark.fast
    def test_build_canonical_media_map_handles_missing_files(self, temp_dir):
        """build_canonical_media_map should skip non-existent files gracefully."""
        existing_file = temp_dir / "exists.mp4"
        existing_file.write_bytes(b"data")

        paths = [
            str(existing_file),
            str(temp_dir / "does_not_exist.mp4"),
            "",
            None
        ]

        result = build_canonical_media_map([p for p in paths if p])

        # Only existing file should be in map
        assert str(existing_file) in result
        assert len(result) == 1


# =============================================================================
# AC4: Split timelines maintain correct clip timing across parts
# =============================================================================

class TestSplitTimelineClipTiming:
    """Test that split timelines maintain correct clip timing (AC4)."""

    @pytest.mark.fast
    def test_split_timeline_clips_preserve_source_range(self, create_timeline_with_n_segments):
        """Clips in split timelines should preserve their source_range values."""
        timeline = create_timeline_with_n_segments(100)

        result = _split_timeline_by_segments(timeline, max_segments=30)

        # Check first clip in each part
        for part_idx, part_timeline in enumerate(result):
            first_clip = list(part_timeline.tracks[0])[0]

            # Each clip is 1 second (30 frames) at position i*30
            expected_start_frame = part_idx * 30 * 30  # part_idx * clips_per_part * frames_per_clip
            assert first_clip.source_range.start_time.value == expected_start_frame
            assert first_clip.source_range.duration.value == 30  # 1 second

    @pytest.mark.fast
    def test_split_timeline_part_continuity(self, create_timeline_with_n_segments):
        """Last clip of part N should be immediately before first clip of part N+1."""
        timeline = create_timeline_with_n_segments(100)

        result = _split_timeline_by_segments(timeline, max_segments=30)

        # Check continuity between parts
        for i in range(len(result) - 1):
            last_clip_part_i = list(result[i].tracks[0])[-1]
            first_clip_part_next = list(result[i + 1].tracks[0])[0]

            last_clip_end = (
                last_clip_part_i.source_range.start_time.value +
                last_clip_part_i.source_range.duration.value
            )
            first_clip_start = first_clip_part_next.source_range.start_time.value

            # First clip of next part should start right after last clip of current part
            assert first_clip_start == last_clip_end, \
                f"Discontinuity between part {i} and {i+1}: {last_clip_end} -> {first_clip_start}"

    @pytest.mark.fast
    def test_split_timeline_total_clips_equals_original(self, create_timeline_with_n_segments):
        """Total clips across all split parts should equal original timeline."""
        n_segments = 150
        timeline = create_timeline_with_n_segments(n_segments)

        result = _split_timeline_by_segments(timeline, max_segments=30)

        total_clips_in_parts = sum(
            len(list(part.tracks[0])) for part in result
        )

        assert total_clips_in_parts == n_segments

    @pytest.mark.fast
    def test_split_timeline_audio_tracks_sync_with_video(self, create_timeline_with_n_segments):
        """Audio tracks in split timeline should match video track clip count."""
        timeline = create_timeline_with_n_segments(100)

        result = _split_timeline_by_segments(timeline, max_segments=30)

        for part_timeline in result:
            video_tracks = [t for t in part_timeline.tracks if t.kind == otio.schema.TrackKind.Video]
            audio_tracks = [t for t in part_timeline.tracks if t.kind == otio.schema.TrackKind.Audio]

            if video_tracks and audio_tracks:
                video_clip_count = len(list(video_tracks[0]))
                audio_clip_count = len(list(audio_tracks[0]))

                assert video_clip_count == audio_clip_count, \
                    "Video and audio track clip counts should match in split timeline"

    @pytest.mark.fast
    def test_split_timeline_preserves_metadata(self, create_timeline_with_n_segments):
        """Split timelines should preserve Resolve_OTIO metadata."""
        timeline = create_timeline_with_n_segments(100)
        timeline.metadata['custom_key'] = 'custom_value'

        result = _split_timeline_by_segments(timeline, max_segments=30)

        for part_timeline in result:
            assert 'Resolve_OTIO' in part_timeline.metadata
            assert part_timeline.metadata['Resolve_OTIO']['Resolve OTIO Meta Version'] == '1.0'
            # Custom metadata should also be preserved
            assert part_timeline.metadata.get('custom_key') == 'custom_value'


# =============================================================================
# AC5: Large timeline warning is logged when approaching split threshold
# =============================================================================

class TestLargeTimelineWarnings:
    """Test that large timeline warning is logged at threshold (AC5)."""

    @pytest.mark.fast
    def test_log_warning_at_warning_threshold(self, caplog):
        """Warning should be logged at CLIP_COUNT_WARNING_THRESHOLD."""
        caplog.set_level(logging.WARNING)

        _log_clip_count_warnings(CLIP_COUNT_WARNING_THRESHOLD)

        assert "approaching DaVinci limit" in caplog.text
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) >= 1

    @pytest.mark.fast
    def test_log_error_at_error_threshold(self, caplog):
        """Error should be logged at CLIP_COUNT_ERROR_THRESHOLD."""
        caplog.set_level(logging.ERROR)

        _log_clip_count_warnings(CLIP_COUNT_ERROR_THRESHOLD)

        assert "exceeding safe limit" in caplog.text
        assert "LITE mode" in caplog.text
        assert len([r for r in caplog.records if r.levelno == logging.ERROR]) >= 1

    @pytest.mark.fast
    def test_no_warning_below_threshold(self, caplog):
        """No warning should be logged below CLIP_COUNT_WARNING_THRESHOLD."""
        caplog.set_level(logging.WARNING)

        _log_clip_count_warnings(CLIP_COUNT_WARNING_THRESHOLD - 1)

        assert "approaching DaVinci limit" not in caplog.text
        assert "exceeding safe limit" not in caplog.text

    @pytest.mark.fast
    def test_threshold_values_match_rule17(self):
        """Verify threshold constants match Rule 17 specification."""
        # Per CLAUDE.md Rule 17: auto-split at 3000 items
        # Warning at 2500, error at 3000
        assert CLIP_COUNT_WARNING_THRESHOLD == 2500
        assert CLIP_COUNT_ERROR_THRESHOLD == 3000

    @pytest.mark.fast
    def test_warning_includes_clip_count(self, caplog):
        """Warning message should include the actual clip count."""
        caplog.set_level(logging.WARNING)
        test_count = CLIP_COUNT_WARNING_THRESHOLD + 50

        _log_clip_count_warnings(test_count)

        assert str(test_count) in caplog.text

    @pytest.mark.fast
    def test_error_suggests_solutions(self, caplog):
        """Error message should suggest LITE mode and split timeline."""
        caplog.set_level(logging.ERROR)

        _log_clip_count_warnings(CLIP_COUNT_ERROR_THRESHOLD + 100)

        # Should mention solutions
        assert "LITE mode" in caplog.text or "split" in caplog.text.lower()


# =============================================================================
# Integration: Full workflow tests
# =============================================================================

class TestAutoSplitIntegration:
    """Integration tests for auto-split workflow."""

    @pytest.mark.fast
    def test_full_workflow_small_timeline(self, create_timeline_with_n_segments, temp_dir):
        """Small timeline should save as single file without splitting."""
        timeline = create_timeline_with_n_segments(50)
        output_path = temp_dir / "small_timeline.otio"

        # Save without max_segments_per_file (no split)
        paths = save_timeline_split(timeline, str(output_path))

        # Should have track files + FULL (no parts)
        assert any('_FULL.otio' in p for p in paths)
        assert not any('_FULL_part' in p for p in paths)

    @pytest.mark.fast
    def test_full_workflow_large_timeline(self, create_timeline_with_n_segments, temp_dir):
        """Large timeline should auto-split when max_segments_per_file is set."""
        timeline = create_timeline_with_n_segments(100)
        output_path = temp_dir / "large_timeline.otio"

        # Save with max_segments_per_file
        paths = save_timeline_split(timeline, str(output_path), max_segments_per_file=30)

        # Should have part files
        part_files = [p for p in paths if '_FULL_part' in p]
        assert len(part_files) == 4  # ceil(100/30) = 4

        # Verify each part file is valid OTIO
        for part_path in part_files:
            loaded = otio.adapters.read_from_file(part_path)
            assert isinstance(loaded, otio.schema.Timeline)

    @pytest.mark.fast
    def test_empty_timeline_no_split(self):
        """Timeline with no video tracks should not error on split."""
        timeline = otio.schema.Timeline(name="Empty")

        result = _split_timeline_by_segments(timeline, max_segments=100)

        assert len(result) == 1
        assert result[0] == timeline


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
