"""
Comprehensive tests for src/otio/reporting.py

Tests segment map JSON generation, timeline statistics printing,
timecode conversion, and entity match type reporting.

Created: 2026-01-09
Test Count: 18 tests
"""

import json
import pytest
import re
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from datetime import datetime

import opentimelineio as otio

# Import module under test
from src.otio.reporting import generate_segment_map, print_timeline_statistics, _calculate_track_coverage

# Import data structures
from src.utils import Match, MatchResult, SRTSegment


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_matches():
    """Create realistic MatchResult objects for testing."""
    matches = []

    # Segment 1: 3 seconds
    vo_seg1 = SRTSegment(
        index=0, start_time=0.0, end_time=3.0,
        text="First voiceover segment", source_file="voiceover.srt"
    )
    vid_seg1 = SRTSegment(
        index=0, start_time=10.0, end_time=13.0,
        text="Video clip 1", source_file="/videos/clip1.mp4"
    )
    match1 = Match(
        voiceover_segment=vo_seg1,
        video_segment=vid_seg1,
        video_scene=None,
        confidence=0.92,
        reasoning='High confidence primary match'
    )

    # Alternative matches for segment 1
    alt1 = Match(
        voiceover_segment=vo_seg1,
        video_segment=SRTSegment(
            index=1, start_time=20.0, end_time=23.0,
            text="Alternative video 1", source_file="/videos/alt1.mp4"
        ),
        video_scene=None,
        confidence=0.85,
        reasoning='Alternative match 1'
    )

    alt2 = Match(
        voiceover_segment=vo_seg1,
        video_segment=SRTSegment(
            index=2, start_time=30.0, end_time=33.0,
            text="Alternative video 2", source_file="/videos/alt2.mp4"
        ),
        video_scene=None,
        confidence=0.78,
        reasoning='Alternative match 2'
    )

    # Secondary matches for segment 1
    sec1 = Match(
        voiceover_segment=vo_seg1,
        video_segment=SRTSegment(
            index=3, start_time=40.0, end_time=43.0,
            text="Secondary video 1", source_file="/videos/sec1.mp4"
        ),
        video_scene=None,
        confidence=0.65,
        reasoning='Secondary match 1'
    )

    matches.append(MatchResult(
        primary_match=match1,
        alternatives=[alt1, alt2],
        secondary_matches=[sec1],
        strategy_matches=[]
    ))

    # Segment 2: 2.5 seconds (no alternatives/secondaries)
    vo_seg2 = SRTSegment(
        index=1, start_time=3.0, end_time=5.5,
        text="Second voiceover segment", source_file="voiceover.srt"
    )
    vid_seg2 = SRTSegment(
        index=4, start_time=50.0, end_time=52.5,
        text="Video clip 2", source_file="/videos/clip2.mp4"
    )
    match2 = Match(
        voiceover_segment=vo_seg2,
        video_segment=vid_seg2,
        video_scene=None,
        confidence=0.88,
        reasoning='Second primary match'
    )

    matches.append(MatchResult(
        primary_match=match2,
        alternatives=[],
        secondary_matches=[],
        strategy_matches=[]
    ))

    return matches


@pytest.fixture
def mock_timeline():
    """Create a realistic OTIO timeline with multiple tracks."""
    timeline = otio.schema.Timeline(name="Test Timeline")

    # V1 - Primary track
    v1 = otio.schema.Track(name="V1 - Primary Video", kind=otio.schema.TrackKind.Video)
    clip1 = otio.schema.Clip(
        name="[S000] folder_clip1.mp4",
        source_range=otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(0, 30),
            duration=otio.opentime.RationalTime(90, 30)  # 3 seconds
        )
    )
    clip1.metadata['segment_index'] = 0

    clip2 = otio.schema.Clip(
        name="[S001] folder_clip2.mp4",
        source_range=otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(0, 30),
            duration=otio.opentime.RationalTime(75, 30)  # 2.5 seconds
        )
    )
    clip2.metadata['segment_index'] = 1

    v1.append(clip1)
    v1.append(clip2)
    timeline.tracks.append(v1)

    # V2 - Alternative track (with gaps)
    v2 = otio.schema.Track(name="V2 - Alternative Video 1", kind=otio.schema.TrackKind.Video)
    clip3 = otio.schema.Clip(
        name="[S000] folder_alt1.mp4",
        source_range=otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(0, 30),
            duration=otio.opentime.RationalTime(90, 30)
        )
    )
    gap1 = otio.schema.Gap(
        source_range=otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(0, 30),
            duration=otio.opentime.RationalTime(75, 30)
        )
    )
    v2.append(clip3)
    v2.append(gap1)
    timeline.tracks.append(v2)

    # V9 - Entity Images (with match_type metadata)
    v9 = otio.schema.Track(name="V9 - Entity Images (Google)", kind=otio.schema.TrackKind.Video)
    entity_clip1 = otio.schema.Clip(
        name="[S000] entity_image1.jpg",
        source_range=otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(0, 30),
            duration=otio.opentime.RationalTime(45, 30)
        )
    )
    entity_clip1.metadata['match_type'] = 'exact'
    entity_clip1.metadata['entity_name'] = 'Barack Obama'

    entity_clip2 = otio.schema.Clip(
        name="[S000] entity_image2.jpg",
        source_range=otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(0, 30),
            duration=otio.opentime.RationalTime(45, 30)
        )
    )
    entity_clip2.metadata['match_type'] = 'semantic'
    entity_clip2.metadata['entity_name'] = 'White House'

    entity_clip3 = otio.schema.Clip(
        name="[S001] entity_image3.jpg",
        source_range=otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(0, 30),
            duration=otio.opentime.RationalTime(75, 30)
        )
    )
    entity_clip3.metadata['match_type'] = 'sticky'
    entity_clip3.metadata['entity_name'] = 'White House'

    v9.append(entity_clip1)
    v9.append(entity_clip2)
    v9.append(entity_clip3)
    timeline.tracks.append(v9)

    # A1 - Audio track
    a1 = otio.schema.Track(name="A1 - Voiceover", kind=otio.schema.TrackKind.Audio)
    audio_clip = otio.schema.Clip(
        name="voiceover.wav",
        source_range=otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(0, 30),
            duration=otio.opentime.RationalTime(165, 30)  # 5.5 seconds total
        )
    )
    a1.append(audio_clip)
    timeline.tracks.append(a1)

    return timeline


# =============================================================================
# TEST: generate_segment_map() - Basic Functionality
# =============================================================================

class TestGenerateSegmentMap:
    """Test segment map JSON generation."""

    @pytest.mark.fast
    def test_basic_segment_map_generation(self, mock_matches, tmp_path):
        """Test that segment map JSON is generated with correct structure."""
        output_path = tmp_path / "timeline.otio"

        json_path = generate_segment_map(
            matches=mock_matches,
            output_path=str(output_path),
            frame_rate=30.0,
            source_srt="voiceover.srt",
            timeline_start_tc="01:00:00:00"
        )

        # Verify file was created
        assert Path(json_path).exists()
        assert json_path.endswith("_segments.json")

        # Load and validate JSON structure
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check top-level structure
        assert 'generated_at' in data
        assert 'source_srt' in data
        assert data['source_srt'] == "voiceover.srt"
        assert data['frame_rate'] == 30.0
        assert data['timeline_start_tc'] == "01:00:00:00"
        assert data['total_segments'] == 2
        assert 'total_frames' in data
        assert 'total_duration_sec' in data
        assert 'segments' in data
        assert len(data['segments']) == 2

    @pytest.mark.fast
    def test_segment_map_timecode_conversion(self, mock_matches, tmp_path):
        """Test that frame counts are correctly converted to timecode."""
        output_path = tmp_path / "timeline.otio"

        json_path = generate_segment_map(
            matches=mock_matches,
            output_path=str(output_path),
            frame_rate=30.0,
            source_srt="voiceover.srt",
            timeline_start_tc="01:00:00:00"
        )

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Segment 0: starts at frame 0 (01:00:00:00), duration 3 seconds = 90 frames
        seg0 = data['segments'][0]
        assert seg0['start_frame'] == 0
        assert seg0['end_frame'] == 90
        assert seg0['start_tc'] == "01:00:00:00"
        assert seg0['end_tc'] == "01:00:03:00"  # 3 seconds = 90 frames at 30fps

        # Segment 1: starts at frame 90, duration 2.5 seconds = 75 frames
        seg1 = data['segments'][1]
        assert seg1['start_frame'] == 90
        assert seg1['end_frame'] == 165
        assert seg1['start_tc'] == "01:00:03:00"
        assert seg1['end_tc'] == "01:00:05:15"  # 5.5 seconds = 165 frames

    @pytest.mark.fast
    def test_segment_map_primary_match_data(self, mock_matches, tmp_path):
        """Test that primary match data is correctly included."""
        output_path = tmp_path / "timeline.otio"

        json_path = generate_segment_map(
            matches=mock_matches,
            output_path=str(output_path),
            frame_rate=30.0
        )

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check segment 0 primary match
        seg0 = data['segments'][0]
        assert seg0['id'] == "S000"
        assert seg0['voiceover_text'] == "First voiceover segment"
        assert seg0['duration_sec'] == 3.0

        v1_clip = seg0['v1_clip']
        assert v1_clip['file'] == "clip1.mp4"
        assert v1_clip['confidence'] == 0.92
        assert v1_clip['source_start'] == 10.0
        assert v1_clip['source_end'] == 13.0

    @pytest.mark.fast
    def test_segment_map_alternatives(self, mock_matches, tmp_path):
        """Test that alternative matches are included in segment map."""
        output_path = tmp_path / "timeline.otio"

        json_path = generate_segment_map(
            matches=mock_matches,
            output_path=str(output_path),
            frame_rate=30.0
        )

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Segment 0 has 2 alternatives
        seg0 = data['segments'][0]
        assert 'alternatives' in seg0
        assert len(seg0['alternatives']) == 2

        # Check alternative 1 (V2)
        alt1 = seg0['alternatives'][0]
        assert alt1['track'] == "V2"
        assert alt1['file'] == "alt1.mp4"
        assert alt1['confidence'] == 0.85

        # Check alternative 2 (V3)
        alt2 = seg0['alternatives'][1]
        assert alt2['track'] == "V3"
        assert alt2['file'] == "alt2.mp4"
        assert alt2['confidence'] == 0.78

        # Segment 1 has no alternatives
        seg1 = data['segments'][1]
        assert 'alternatives' not in seg1

    @pytest.mark.fast
    def test_segment_map_secondary_matches(self, mock_matches, tmp_path):
        """Test that secondary matches are included in segment map."""
        output_path = tmp_path / "timeline.otio"

        json_path = generate_segment_map(
            matches=mock_matches,
            output_path=str(output_path),
            frame_rate=30.0
        )

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Segment 0 has 1 secondary match
        seg0 = data['segments'][0]
        assert 'secondary' in seg0
        assert len(seg0['secondary']) == 1

        # Check secondary 1 (V4)
        sec1 = seg0['secondary'][0]
        assert sec1['track'] == "V4"
        assert sec1['file'] == "sec1.mp4"
        assert sec1['confidence'] == 0.65

        # Segment 1 has no secondaries
        seg1 = data['segments'][1]
        assert 'secondary' not in seg1

    @pytest.mark.fast
    def test_segment_map_custom_frame_rate(self, mock_matches, tmp_path):
        """Test segment map generation with non-standard frame rate."""
        output_path = tmp_path / "timeline.otio"

        json_path = generate_segment_map(
            matches=mock_matches,
            output_path=str(output_path),
            frame_rate=24.0,  # Film frame rate
            timeline_start_tc="00:00:00:00"
        )

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert data['frame_rate'] == 24.0
        assert data['timeline_start_tc'] == "00:00:00:00"

        # 3 seconds at 24fps = 72 frames
        seg0 = data['segments'][0]
        assert seg0['end_frame'] == 72
        assert seg0['end_tc'] == "00:00:03:00"

    @pytest.mark.fast
    def test_segment_map_file_naming(self, tmp_path):
        """Test that output file is named correctly."""
        # Test with .otio extension
        output_path1 = tmp_path / "timeline.otio"
        json_path1 = generate_segment_map([], str(output_path1), frame_rate=30.0)
        assert json_path1.endswith("timeline_segments.json")

        # Test with .xml extension
        output_path2 = tmp_path / "timeline.xml"
        json_path2 = generate_segment_map([], str(output_path2), frame_rate=30.0)
        assert json_path2.endswith("timeline_segments.json")

        # Test with no extension
        output_path3 = tmp_path / "timeline"
        json_path3 = generate_segment_map([], str(output_path3), frame_rate=30.0)
        assert json_path3.endswith("timeline_segments.json")


# =============================================================================
# TEST: print_timeline_statistics() - Output Formatting
# =============================================================================

class TestPrintTimelineStatistics:
    """Test timeline statistics printing."""

    @pytest.mark.fast
    def test_basic_statistics_output(self, mock_timeline, capsys):
        """Test that statistics are printed with correct structure."""
        print_timeline_statistics(mock_timeline)

        captured = capsys.readouterr()
        output = captured.out

        # Check header
        assert "OTIO TIMELINE STATISTICS" in output
        assert "Test Timeline" in output

        # Check track counts
        assert "Video Tracks: 3" in output  # V1, V2, V9
        assert "Audio Tracks: 1" in output  # A1

        # Check sections present
        assert "TRACK BREAKDOWN" in output
        assert "ENTITY MATCHING" in output
        assert "SEGMENT ID COVERAGE" in output
        assert "QUALITY CHECKLIST" in output

    @pytest.mark.fast
    def test_track_breakdown_calculations(self, mock_timeline, capsys):
        """Test that clip/gap counts and coverage are calculated correctly."""
        print_timeline_statistics(mock_timeline)

        captured = capsys.readouterr()
        output = captured.out

        # V1 has 2 clips, 0 gaps, 100% coverage
        assert "V1 - Primary Video" in output
        assert re.search(r"V1.*2.*0.*100\.0%", output)

        # V2 has 1 clip, 1 gap, ~54.5% coverage (90 frames clip / 165 total)
        assert "V2 - Alternative Video" in output
        assert re.search(r"V2.*1.*1", output)

        # V9 has 3 clips, 0 gaps, 100% coverage
        assert "V9 - Entity Images" in output
        assert re.search(r"V9.*3.*0.*100\.0%", output)

    @pytest.mark.fast
    def test_entity_matching_statistics(self, mock_timeline, capsys):
        """Test entity match type reporting (exact/semantic/sticky)."""
        print_timeline_statistics(mock_timeline)

        captured = capsys.readouterr()
        output = captured.out

        # V9 has 3 entity clips: 1 exact, 1 semantic, 1 sticky
        assert "Total entity clips: 3" in output
        assert "Exact matches:" in output
        assert "1" in output  # 1 exact match
        assert "Semantic matches:" in output
        assert "1" in output  # 1 semantic match
        assert "Sticky (carried):" in output
        assert "1" in output  # 1 sticky match

    @pytest.mark.fast
    def test_segment_id_extraction(self, mock_timeline, capsys):
        """Test segment ID extraction from clip names."""
        print_timeline_statistics(mock_timeline)

        captured = capsys.readouterr()
        output = captured.out

        # All clips have segment IDs [S000] or [S001]
        assert "Clips with segment IDs:" in output
        assert "Unique segments:" in output
        assert "Segment range:" in output
        assert "S000 - S001" in output

    @pytest.mark.fast
    def test_quality_checklist(self, mock_timeline, capsys):
        """Test quality checklist items."""
        print_timeline_statistics(mock_timeline)

        captured = capsys.readouterr()
        output = captured.out

        # All checks should pass for mock timeline
        assert "[✓] V1 Primary track populated" in output
        assert "[✓] Segment IDs present" in output
        assert "[✓] Multiple video tracks" in output
        assert "[✓] Audio track present" in output
        assert "[✓] V9 Entity Images track" in output

    @pytest.mark.fast
    def test_empty_timeline(self, capsys):
        """Test statistics for empty timeline."""
        timeline = otio.schema.Timeline(name="Empty Timeline")

        print_timeline_statistics(timeline)

        captured = capsys.readouterr()
        output = captured.out

        assert "Empty Timeline" in output
        assert "Video Tracks: 0" in output
        assert "Audio Tracks: 0" in output
        assert "No entity clips found" in output

    @pytest.mark.fast
    def test_timeline_without_entity_tracks(self, capsys):
        """Test statistics for timeline without entity tracks."""
        timeline = otio.schema.Timeline(name="No Entity Timeline")

        # Add only V1 track
        v1 = otio.schema.Track(name="V1 - Primary Video", kind=otio.schema.TrackKind.Video)
        clip = otio.schema.Clip(
            name="clip1.mp4",
            source_range=otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(0, 30),
                duration=otio.opentime.RationalTime(90, 30)
            )
        )
        v1.append(clip)
        timeline.tracks.append(v1)

        print_timeline_statistics(timeline)

        captured = capsys.readouterr()
        output = captured.out

        assert "No entity clips found (V9/V10 empty or not provided)" in output
        assert "[✗] V9 Entity Images track" in output
        assert "[✗] V10 Stock Videos track" in output


# =============================================================================
# TEST: Track Coverage Reporting
# =============================================================================

class TestTrackCoverageReporting:
    """Test track coverage statistics in segment map JSON."""

    @pytest.mark.fast
    def test_track_coverage_present_in_output(self, mock_matches, tmp_path):
        """Test that track_coverage dict is included in segment map."""
        output_path = tmp_path / "timeline.otio"

        json_path = generate_segment_map(
            matches=mock_matches,
            output_path=str(output_path),
            frame_rate=30.0
        )

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert 'track_coverage' in data
        assert isinstance(data['track_coverage'], dict)

    @pytest.mark.fast
    def test_track_coverage_includes_v1_to_v10(self, mock_matches, tmp_path):
        """Test that all V1-V10 tracks are included in coverage report."""
        output_path = tmp_path / "timeline.otio"

        json_path = generate_segment_map(
            matches=mock_matches,
            output_path=str(output_path),
            frame_rate=30.0
        )

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        track_coverage = data['track_coverage']

        # Check all V1-V10 tracks present
        expected_tracks = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10']
        for track in expected_tracks:
            assert track in track_coverage, f"Track {track} missing from coverage"

    @pytest.mark.fast
    def test_track_coverage_contains_required_fields(self, mock_matches, tmp_path):
        """Test that each track has clip_count, gap_count, and coverage_percent (or unavailable status)."""
        output_path = tmp_path / "timeline.otio"

        json_path = generate_segment_map(
            matches=mock_matches,
            output_path=str(output_path),
            frame_rate=30.0
        )

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for track_id, stats in data['track_coverage'].items():
            assert 'description' in stats, f"{track_id} missing description"
            if stats.get('status') == 'unavailable':
                # V9/V10 without entity data show unavailable
                continue
            assert 'clip_count' in stats, f"{track_id} missing clip_count"
            assert 'gap_count' in stats, f"{track_id} missing gap_count"
            assert 'coverage_percent' in stats, f"{track_id} missing coverage_percent"

    @pytest.mark.fast
    def test_v1_coverage_calculation(self, mock_matches, tmp_path):
        """Test V1 primary track coverage is calculated correctly."""
        output_path = tmp_path / "timeline.otio"

        json_path = generate_segment_map(
            matches=mock_matches,
            output_path=str(output_path),
            frame_rate=30.0
        )

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        v1_stats = data['track_coverage']['V1']

        # mock_matches has 2 segments, both with primary matches (no gaps)
        assert v1_stats['clip_count'] == 2
        assert v1_stats['gap_count'] == 0
        assert v1_stats['coverage_percent'] == 100.0

    @pytest.mark.fast
    def test_v2_v3_alternatives_coverage(self, mock_matches, tmp_path):
        """Test V2-V3 alternative track coverage calculation."""
        output_path = tmp_path / "timeline.otio"

        json_path = generate_segment_map(
            matches=mock_matches,
            output_path=str(output_path),
            frame_rate=30.0
        )

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Segment 0 has 2 alternatives, Segment 1 has 0
        # Total duration: 3.0 + 2.5 = 5.5 seconds

        # V2: 1 clip (segment 0 has alt 0), 1 gap (segment 1 has no alt)
        v2_stats = data['track_coverage']['V2']
        assert v2_stats['clip_count'] == 1
        assert v2_stats['gap_count'] == 1

        # V3: 1 clip (segment 0 has alt 1), 1 gap (segment 1 has no alt)
        v3_stats = data['track_coverage']['V3']
        assert v3_stats['clip_count'] == 1
        assert v3_stats['gap_count'] == 1

        # Coverage should be ~54.5% (3.0/5.5 * 100)
        assert 54.0 <= v2_stats['coverage_percent'] <= 55.0
        assert 54.0 <= v3_stats['coverage_percent'] <= 55.0

    @pytest.mark.fast
    def test_secondary_tracks_coverage(self, mock_matches, tmp_path):
        """Test V4-V6 secondary track coverage calculation."""
        output_path = tmp_path / "timeline.otio"

        json_path = generate_segment_map(
            matches=mock_matches,
            output_path=str(output_path),
            frame_rate=30.0
        )

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Segment 0 has 1 secondary, Segment 1 has 0
        # V4: 1 clip (segment 0), 1 gap (segment 1)
        v4_stats = data['track_coverage']['V4']
        assert v4_stats['clip_count'] == 1
        assert v4_stats['gap_count'] == 1
        assert 54.0 <= v4_stats['coverage_percent'] <= 55.0

        # V5, V6: 0 clips, 2 gaps (no secondary matches at these indices)
        v5_stats = data['track_coverage']['V5']
        assert v5_stats['clip_count'] == 0
        assert v5_stats['gap_count'] == 2
        assert v5_stats['coverage_percent'] == 0.0

        v6_stats = data['track_coverage']['V6']
        assert v6_stats['clip_count'] == 0
        assert v6_stats['gap_count'] == 2
        assert v6_stats['coverage_percent'] == 0.0

    @pytest.mark.fast
    def test_empty_matches_track_coverage(self, tmp_path):
        """Test track coverage with empty matches returns empty dict."""
        output_path = tmp_path / "timeline.otio"

        json_path = generate_segment_map(
            matches=[],
            output_path=str(output_path),
            frame_rate=30.0
        )

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert data['track_coverage'] == {}

    @pytest.mark.fast
    def test_track_coverage_duration_calculations(self, mock_matches, tmp_path):
        """Test that clip and gap durations are calculated correctly."""
        output_path = tmp_path / "timeline.otio"

        json_path = generate_segment_map(
            matches=mock_matches,
            output_path=str(output_path),
            frame_rate=30.0
        )

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        v1_stats = data['track_coverage']['V1']

        # Total V1 clip duration should be 3.0 + 2.5 = 5.5 seconds
        assert v1_stats['clip_duration_sec'] == 5.5
        assert v1_stats['gap_duration_sec'] == 0.0

    @pytest.mark.fast
    def test_has_gap_affects_v1_coverage(self, tmp_path):
        """Test that has_gap=True segments count as gaps on V1."""
        # Create a match with has_gap=True
        vo_seg = SRTSegment(
            index=0, start_time=0.0, end_time=3.0,
            text="Segment text", source_file="voiceover.srt"
        )
        vid_seg = SRTSegment(
            index=0, start_time=10.0, end_time=13.0,
            text="Video", source_file="/videos/clip.mp4"
        )
        match = Match(
            voiceover_segment=vo_seg,
            video_segment=vid_seg,
            video_scene=None,
            confidence=0.3,  # Low confidence
            reasoning='Low confidence match'
        )
        matches = [MatchResult(
            primary_match=match,
            alternatives=[],
            secondary_matches=[],
            strategy_matches=[],
            has_gap=True,  # This segment has a gap
            gap_reason="No good match found"
        )]

        output_path = tmp_path / "timeline.otio"
        json_path = generate_segment_map(matches, str(output_path), frame_rate=30.0)

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        v1_stats = data['track_coverage']['V1']
        assert v1_stats['clip_count'] == 0
        assert v1_stats['gap_count'] == 1
        assert v1_stats['coverage_percent'] == 0.0


# =============================================================================
# TEST: Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.fast
    def test_empty_matches_list(self, tmp_path):
        """Test segment map generation with empty matches."""
        output_path = tmp_path / "timeline.otio"

        json_path = generate_segment_map(
            matches=[],
            output_path=str(output_path),
            frame_rate=30.0
        )

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert data['total_segments'] == 0
        assert data['total_frames'] == 0
        assert data['total_duration_sec'] == 0.0
        assert len(data['segments']) == 0

    @pytest.mark.fast
    def test_single_segment(self, tmp_path):
        """Test segment map with single segment."""
        vo_seg = SRTSegment(
            index=0, start_time=0.0, end_time=2.0,
            text="Single segment", source_file="voiceover.srt"
        )
        vid_seg = SRTSegment(
            index=0, start_time=0.0, end_time=2.0,
            text="Video", source_file="/videos/clip.mp4"
        )
        match = Match(
            voiceover_segment=vo_seg,
            video_segment=vid_seg,
            video_scene=None,
            confidence=0.9,
            reasoning='Single match'
        )
        matches = [MatchResult(
            primary_match=match,
            alternatives=[],
            secondary_matches=[],
            strategy_matches=[]
        )]

        output_path = tmp_path / "timeline.otio"
        json_path = generate_segment_map(matches, str(output_path), frame_rate=30.0)

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert data['total_segments'] == 1
        assert data['total_frames'] == 60  # 2 seconds * 30fps
        assert len(data['segments']) == 1


# =============================================================================
# TEST: V9/V10 Entity Track Coverage (US-56-010)
# =============================================================================

class TestEntityTrackCoverage:
    """Test V9/V10 track coverage reflects actual entity data."""

    @pytest.mark.fast
    def test_v9_coverage_with_entity_images(self):
        """Test V9 shows actual clip counts when entity_images provided with 5 entities."""
        # Create 3 segments
        matches = []
        for i in range(3):
            vo_seg = SRTSegment(
                index=i, start_time=float(i * 2), end_time=float(i * 2 + 2),
                text=f"Segment {i}", source_file="voiceover.srt"
            )
            vid_seg = SRTSegment(
                index=i, start_time=float(i * 10), end_time=float(i * 10 + 2),
                text=f"Video {i}", source_file=f"/videos/clip{i}.mp4"
            )
            match = Match(
                voiceover_segment=vo_seg, video_segment=vid_seg,
                video_scene=None, confidence=0.9, reasoning='Match'
            )
            matches.append(MatchResult(
                primary_match=match, alternatives=[],
                secondary_matches=[], strategy_matches=[]
            ))

        # Create 5 entity image results covering segments 0 and 1
        entity_images = {}
        for j in range(5):
            entity = Mock()
            entity.entity_name = f"Entity {j}"
            entity.images = [f"/images/entity{j}_img1.jpg", f"/images/entity{j}_img2.jpg"]
            # First 3 entities cover segment 0, last 2 cover segment 1
            entity.segment_indices = [0] if j < 3 else [1]
            entity_images[f"Entity {j}"] = entity

        stats = _calculate_track_coverage(matches, 30.0, entity_images=entity_images)

        # V9 should have clips for segments 0 and 1, gap for segment 2
        assert stats["V9"]["clip_count"] == 2
        assert stats["V9"]["gap_count"] == 1
        assert stats["V9"]["clip_count"] > 0  # Acceptance criterion
        assert stats["V9"]["coverage_percent"] > 0

    @pytest.mark.fast
    def test_v10_coverage_with_entity_videos(self):
        """Test V10 shows actual clip counts when entity_videos provided."""
        matches = []
        for i in range(2):
            vo_seg = SRTSegment(
                index=i, start_time=float(i * 3), end_time=float(i * 3 + 3),
                text=f"Segment {i}", source_file="voiceover.srt"
            )
            vid_seg = SRTSegment(
                index=i, start_time=float(i * 10), end_time=float(i * 10 + 3),
                text=f"Video {i}", source_file=f"/videos/clip{i}.mp4"
            )
            match = Match(
                voiceover_segment=vo_seg, video_segment=vid_seg,
                video_scene=None, confidence=0.9, reasoning='Match'
            )
            matches.append(MatchResult(
                primary_match=match, alternatives=[],
                secondary_matches=[], strategy_matches=[]
            ))

        entity_videos = {
            "Entity A": Mock(videos=["/videos/stock1.mp4"], segment_indices=[0, 1]),
        }

        stats = _calculate_track_coverage(matches, 30.0, entity_videos=entity_videos)

        assert stats["V10"]["clip_count"] == 2
        assert stats["V10"]["gap_count"] == 0
        assert stats["V10"]["coverage_percent"] == 100.0

    @pytest.mark.fast
    def test_v9_v10_unavailable_without_entity_data(self):
        """Test V9/V10 report unavailable status when entity data not provided."""
        vo_seg = SRTSegment(
            index=0, start_time=0.0, end_time=3.0,
            text="Segment", source_file="voiceover.srt"
        )
        vid_seg = SRTSegment(
            index=0, start_time=10.0, end_time=13.0,
            text="Video", source_file="/videos/clip.mp4"
        )
        match = Match(
            voiceover_segment=vo_seg, video_segment=vid_seg,
            video_scene=None, confidence=0.9, reasoning='Match'
        )
        matches = [MatchResult(
            primary_match=match, alternatives=[],
            secondary_matches=[], strategy_matches=[]
        )]

        # Call without entity data (default None)
        stats = _calculate_track_coverage(matches, 30.0)

        # V9 and V10 should be unavailable, not 100% gap
        assert stats["V9"]["status"] == "unavailable"
        assert stats["V10"]["status"] == "unavailable"
        assert "clip_count" not in stats["V9"]
        assert "gap_count" not in stats["V9"]
        assert "clip_count" not in stats["V10"]
        assert "gap_count" not in stats["V10"]

    @pytest.mark.fast
    def test_v9_entity_images_no_images_for_segment(self):
        """Test V9 gap when entity exists but has no images for a segment."""
        vo_seg = SRTSegment(
            index=0, start_time=0.0, end_time=2.0,
            text="Segment", source_file="voiceover.srt"
        )
        vid_seg = SRTSegment(
            index=0, start_time=10.0, end_time=12.0,
            text="Video", source_file="/videos/clip.mp4"
        )
        match = Match(
            voiceover_segment=vo_seg, video_segment=vid_seg,
            video_scene=None, confidence=0.9, reasoning='Match'
        )
        matches = [MatchResult(
            primary_match=match, alternatives=[],
            secondary_matches=[], strategy_matches=[]
        )]

        # Entity covers segment 0 but has empty images list
        entity_images = {
            "Entity A": Mock(images=[], segment_indices=[0]),
        }

        stats = _calculate_track_coverage(matches, 30.0, entity_images=entity_images)

        assert stats["V9"]["clip_count"] == 0
        assert stats["V9"]["gap_count"] == 1

    @pytest.mark.fast
    def test_generate_segment_map_with_entity_data(self, tmp_path):
        """Test that generate_segment_map passes entity data to coverage calc."""
        vo_seg = SRTSegment(
            index=0, start_time=0.0, end_time=2.0,
            text="Segment", source_file="voiceover.srt"
        )
        vid_seg = SRTSegment(
            index=0, start_time=10.0, end_time=12.0,
            text="Video", source_file="/videos/clip.mp4"
        )
        match = Match(
            voiceover_segment=vo_seg, video_segment=vid_seg,
            video_scene=None, confidence=0.9, reasoning='Match'
        )
        matches = [MatchResult(
            primary_match=match, alternatives=[],
            secondary_matches=[], strategy_matches=[]
        )]

        entity_images = {
            "Entity A": Mock(images=["/img/a.jpg"], segment_indices=[0]),
        }

        output_path = tmp_path / "timeline.otio"
        json_path = generate_segment_map(
            matches=matches,
            output_path=str(output_path),
            frame_rate=30.0,
            entity_images=entity_images
        )

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # V9 should show clip (not gap) since entity data provided
        assert data['track_coverage']['V9']['clip_count'] == 1
        assert data['track_coverage']['V9']['gap_count'] == 0

        # V10 should be unavailable since no entity_videos provided
        assert data['track_coverage']['V10']['status'] == 'unavailable'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
