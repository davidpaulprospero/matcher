"""
Unit tests for src/otio/export.py

Tests OTIO export functions including:
- save_timeline: Basic OTIO file saving
- save_timeline_split: Split timeline into track-specific files
- save_timeline_as_edl: EDL marker export for DaVinci Resolve
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSaveTimeline:
    """Tests for save_timeline function."""

    def test_save_timeline_basic(self, tmp_path):
        """Test basic timeline saving."""
        import opentimelineio as otio
        from src.otio.export import save_timeline

        timeline = otio.schema.Timeline(name="Test Timeline")
        output_path = str(tmp_path / "test.otio")

        save_timeline(timeline, output_path)

        assert Path(output_path).exists()
        # Verify we can read it back
        loaded = otio.adapters.read_from_file(output_path)
        assert loaded.name == "Test Timeline"

    def test_save_timeline_with_tracks(self, tmp_path):
        """Test saving timeline with video and audio tracks."""
        import opentimelineio as otio
        from src.otio.export import save_timeline

        timeline = otio.schema.Timeline(name="Multi-Track")
        video_track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
        audio_track = otio.schema.Track(name="A1", kind=otio.schema.TrackKind.Audio)
        timeline.tracks.append(video_track)
        timeline.tracks.append(audio_track)

        output_path = str(tmp_path / "multi_track.otio")
        save_timeline(timeline, output_path)

        loaded = otio.adapters.read_from_file(output_path)
        assert len(loaded.tracks) == 2


class TestSaveTimelineSplit:
    """Tests for save_timeline_split function."""

    @pytest.fixture
    def sample_timeline(self):
        """Create a sample timeline with multiple tracks."""
        import opentimelineio as otio

        timeline = otio.schema.Timeline(name="Test Timeline")
        timeline.global_start_time = otio.opentime.RationalTime(108000, 30.0)
        timeline.metadata['Resolve_OTIO'] = {'Resolve OTIO Meta Version': '1.0'}

        # Add video tracks V1-V3
        for i in range(3):
            track = otio.schema.Track(
                name=f"V{i+1} - Primary" if i == 0 else f"V{i+1} - Alt {i}",
                kind=otio.schema.TrackKind.Video
            )
            # Add a clip
            clip = otio.schema.Clip(
                name=f"Clip_{i}",
                source_range=otio.opentime.TimeRange(
                    otio.opentime.RationalTime(0, 30),
                    otio.opentime.RationalTime(90, 30)
                )
            )
            track.append(clip)
            timeline.tracks.append(track)

        # Add audio tracks A1-A3
        for i in range(3):
            track = otio.schema.Track(
                name=f"A{i+1}",
                kind=otio.schema.TrackKind.Audio
            )
            timeline.tracks.append(track)

        # Add voiceover track A8
        vo_track = otio.schema.Track(
            name="A8 - Voiceover",
            kind=otio.schema.TrackKind.Audio
        )
        timeline.tracks.append(vo_track)

        return timeline

    def test_save_timeline_split_creates_files(self, tmp_path, sample_timeline):
        """Test that split creates expected files."""
        from src.otio.export import save_timeline_split

        output_path = str(tmp_path / "output.otio")
        paths = save_timeline_split(sample_timeline, output_path)

        # Should create: V1, V2, V3, A8 voiceover, FULL
        assert len(paths) >= 4
        assert all(Path(p).exists() for p in paths)

    def test_save_timeline_split_full_timeline(self, tmp_path, sample_timeline):
        """Test that FULL timeline file is created."""
        from src.otio.export import save_timeline_split

        output_path = str(tmp_path / "project.otio")
        paths = save_timeline_split(sample_timeline, output_path)

        # Find FULL file
        full_path = [p for p in paths if '_FULL.otio' in p]
        assert len(full_path) == 1
        assert Path(full_path[0]).exists()

    def test_save_timeline_split_track_files(self, tmp_path, sample_timeline):
        """Test individual track files are created."""
        import opentimelineio as otio
        from src.otio.export import save_timeline_split

        output_path = str(tmp_path / "project.otio")
        paths = save_timeline_split(sample_timeline, output_path)

        # Check V1 track file exists
        v1_files = [p for p in paths if '_V1_' in p]
        assert len(v1_files) == 1

        # Load and verify V1 file
        loaded = otio.adapters.read_from_file(v1_files[0])
        assert len(loaded.tracks) == 1
        assert loaded.tracks[0].enabled is True

    def test_save_timeline_split_voiceover_track(self, tmp_path, sample_timeline):
        """Test voiceover track file is created."""
        from src.otio.export import save_timeline_split

        output_path = str(tmp_path / "project.otio")
        paths = save_timeline_split(sample_timeline, output_path)

        # Find voiceover file
        vo_files = [p for p in paths if '_A8_voiceover' in p]
        assert len(vo_files) == 1

    def test_save_timeline_split_preserves_frame_rate(self, tmp_path, sample_timeline):
        """Test that frame rate is preserved in split files."""
        import opentimelineio as otio
        from src.otio.export import save_timeline_split

        output_path = str(tmp_path / "project.otio")
        paths = save_timeline_split(sample_timeline, output_path)

        # Load V1 and check frame rate
        v1_files = [p for p in paths if '_V1_' in p]
        loaded = otio.adapters.read_from_file(v1_files[0])
        assert loaded.global_start_time.rate == 30.0

    def test_save_timeline_split_resolve_metadata(self, tmp_path, sample_timeline):
        """Test DaVinci Resolve metadata is added."""
        import opentimelineio as otio
        from src.otio.export import save_timeline_split

        output_path = str(tmp_path / "project.otio")
        paths = save_timeline_split(sample_timeline, output_path)

        # Load V1 and check metadata
        v1_files = [p for p in paths if '_V1_' in p]
        loaded = otio.adapters.read_from_file(v1_files[0])
        assert 'Resolve_OTIO' in loaded.metadata

    def test_save_timeline_split_default_frame_rate(self, tmp_path):
        """Test default frame rate when not specified."""
        import opentimelineio as otio
        from src.otio.export import save_timeline_split

        # Create timeline without global_start_time
        timeline = otio.schema.Timeline(name="No Start Time")
        track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
        timeline.tracks.append(track)

        output_path = str(tmp_path / "project.otio")
        paths = save_timeline_split(timeline, output_path)

        # Should still work with default 30fps
        assert len(paths) >= 1

    def test_save_timeline_split_safe_name(self, tmp_path):
        """Test safe_name function handles special characters."""
        import opentimelineio as otio
        from src.otio.export import save_timeline_split

        timeline = otio.schema.Timeline(name="Test")
        track = otio.schema.Track(
            name="V1 - Special/Chars-Here",
            kind=otio.schema.TrackKind.Video
        )
        timeline.tracks.append(track)

        output_path = str(tmp_path / "project.otio")
        paths = save_timeline_split(timeline, output_path)

        # Should create file with sanitized name
        v1_files = [p for p in paths if '_V1_' in p]
        assert len(v1_files) == 1
        # No slashes or dashes in filename
        assert '/' not in Path(v1_files[0]).name

    def test_save_timeline_split_clip_count_logging(self, tmp_path, sample_timeline):
        """Test that clip counts are calculated correctly."""
        from src.otio.export import save_timeline_split

        output_path = str(tmp_path / "project.otio")
        # Should not raise any errors
        paths = save_timeline_split(sample_timeline, output_path)
        assert len(paths) > 0


class TestSaveTimelineAsEdl:
    """Tests for save_timeline_as_edl function."""

    @pytest.fixture
    def mock_voiceover_segment(self):
        """Create a mock voiceover segment."""
        segment = Mock()
        segment.start_time = 0.0
        segment.end_time = 5.0
        segment.text = "This is a test voiceover segment for testing purposes."
        return segment

    @pytest.fixture
    def mock_match(self, mock_voiceover_segment):
        """Create a mock match object."""
        match = Mock()
        match.confidence = 0.85
        match.voiceover_segment = mock_voiceover_segment
        return match

    @pytest.fixture
    def mock_match_result(self, mock_match):
        """Create a mock match result."""
        result = Mock()
        result.primary_match = mock_match
        return result

    def test_save_timeline_as_edl_creates_file(self, tmp_path, mock_match_result):
        """Test EDL file is created."""
        from src.otio.export import save_timeline_as_edl

        output_path = str(tmp_path / "markers.edl")
        result = save_timeline_as_edl([mock_match_result], output_path)

        assert Path(result).exists()
        assert result.endswith('.edl')

    def test_save_timeline_as_edl_content(self, tmp_path, mock_match_result):
        """Test EDL file contains expected content."""
        from src.otio.export import save_timeline_as_edl

        output_path = str(tmp_path / "markers.edl")
        save_timeline_as_edl([mock_match_result], output_path)

        content = Path(output_path).with_suffix('.edl').read_text()
        assert "TITLE: Matched Footage Markers" in content
        assert "FCM: NON-DROP FRAME" in content

    def test_save_timeline_as_edl_marker_entry(self, tmp_path, mock_match_result):
        """Test EDL contains marker entry."""
        from src.otio.export import save_timeline_as_edl

        output_path = str(tmp_path / "markers.edl")
        save_timeline_as_edl([mock_match_result], output_path)

        content = Path(output_path).with_suffix('.edl').read_text()
        # Should have marker entry line
        assert "001  BL       V     C" in content
        assert "FROM CLIP NAME" in content

    def test_save_timeline_as_edl_truncates_long_text(self, tmp_path):
        """Test that long marker names are truncated."""
        from src.otio.export import save_timeline_as_edl

        segment = Mock()
        segment.start_time = 0.0
        segment.end_time = 5.0
        segment.text = "A" * 100  # Very long text

        match = Mock()
        match.confidence = 0.8
        match.voiceover_segment = segment

        result = Mock()
        result.primary_match = match

        output_path = str(tmp_path / "markers.edl")
        save_timeline_as_edl([result], output_path)

        content = Path(output_path).with_suffix('.edl').read_text()
        # Text should be truncated to 40 chars + "..."
        assert "A" * 40 + "..." in content

    def test_save_timeline_as_edl_with_entities(self, tmp_path, mock_match_result):
        """Test EDL with entity markers."""
        from src.otio.export import save_timeline_as_edl

        entities = [
            {'name': 'Person A', 'position_sec': 10.0},
            {'name': 'Place B', 'position_sec': 25.0}
        ]

        output_path = str(tmp_path / "markers.edl")
        save_timeline_as_edl([mock_match_result], output_path, entities=entities)

        content = Path(output_path).with_suffix('.edl').read_text()
        assert "Entity: Person A" in content
        assert "Entity: Place B" in content
        # Entity markers should use Pink color
        assert "ResolveColorPink" in content

    def test_save_timeline_as_edl_confidence_colors(self, tmp_path):
        """Test that confidence affects marker colors."""
        from src.otio.export import save_timeline_as_edl

        # High confidence
        segment = Mock()
        segment.start_time = 0.0
        segment.end_time = 2.0
        segment.text = "High confidence"

        match = Mock()
        match.confidence = 0.95
        match.voiceover_segment = segment

        result = Mock()
        result.primary_match = match

        output_path = str(tmp_path / "markers.edl")
        save_timeline_as_edl([result], output_path)

        content = Path(output_path).with_suffix('.edl').read_text()
        # Should contain marker color information
        assert "|C:ResolveColor" in content

    def test_save_timeline_as_edl_timecode_format(self, tmp_path, mock_match_result):
        """Test timecode formatting."""
        from src.otio.export import save_timeline_as_edl

        output_path = str(tmp_path / "markers.edl")
        save_timeline_as_edl([mock_match_result], output_path, timeline_start_tc="01:00:00:00")

        content = Path(output_path).with_suffix('.edl').read_text()
        # Timecode should start at 01:00:00:00
        assert "01:00:00:00" in content

    def test_save_timeline_as_edl_custom_frame_rate(self, tmp_path, mock_match_result):
        """Test custom frame rate."""
        from src.otio.export import save_timeline_as_edl

        output_path = str(tmp_path / "markers.edl")
        save_timeline_as_edl([mock_match_result], output_path, frame_rate=24.0)

        # Should complete without error
        assert Path(output_path).with_suffix('.edl').exists()

    def test_save_timeline_as_edl_multiple_matches(self, tmp_path):
        """Test EDL with multiple match results."""
        from src.otio.export import save_timeline_as_edl

        matches = []
        for i in range(5):
            segment = Mock()
            segment.start_time = i * 5.0
            segment.end_time = (i + 1) * 5.0
            segment.text = f"Segment {i+1}"

            match = Mock()
            match.confidence = 0.5 + i * 0.1
            match.voiceover_segment = segment

            result = Mock()
            result.primary_match = match
            matches.append(result)

        output_path = str(tmp_path / "markers.edl")
        save_timeline_as_edl(matches, output_path)

        content = Path(output_path).with_suffix('.edl').read_text()
        # Should have 5 marker entries
        assert "005  BL" in content
        assert "Segment 5" in content

    def test_save_timeline_as_edl_empty_entities(self, tmp_path, mock_match_result):
        """Test EDL with empty entities list."""
        from src.otio.export import save_timeline_as_edl

        output_path = str(tmp_path / "markers.edl")
        save_timeline_as_edl([mock_match_result], output_path, entities=[])

        assert Path(output_path).with_suffix('.edl').exists()

    def test_save_timeline_as_edl_duration_frames(self, tmp_path):
        """Test frame calculation from duration."""
        from src.otio.export import save_timeline_as_edl

        segment = Mock()
        segment.start_time = 0.0
        segment.end_time = 10.0  # 10 second segment = 300 frames at 30fps
        segment.text = "Ten second segment"

        match = Mock()
        match.confidence = 0.75
        match.voiceover_segment = segment

        result = Mock()
        result.primary_match = match

        output_path = str(tmp_path / "markers.edl")
        save_timeline_as_edl([result], output_path)

        # File should be created
        assert Path(output_path).with_suffix('.edl').exists()


class TestEdlTimecodeConversion:
    """Tests for internal timecode conversion in EDL export."""

    def test_frames_to_tc_start_offset(self, tmp_path):
        """Test timecode calculation includes start offset."""
        from src.otio.export import save_timeline_as_edl

        segment = Mock()
        segment.start_time = 0.0
        segment.end_time = 1.0
        segment.text = "First segment"

        match = Mock()
        match.confidence = 0.8
        match.voiceover_segment = segment

        result = Mock()
        result.primary_match = match

        # Start at 02:00:00:00 instead of default 01:00:00:00
        output_path = str(tmp_path / "markers.edl")
        save_timeline_as_edl([result], output_path, timeline_start_tc="02:00:00:00")

        content = Path(output_path).with_suffix('.edl').read_text()
        # First marker should be at 02:00:00:00
        assert "02:00:00:00" in content

    def test_frames_to_tc_rollover(self, tmp_path):
        """Test timecode handles minute/hour rollover."""
        from src.otio.export import save_timeline_as_edl

        # Create a segment that would be at 01:01:30:00 (90 seconds in)
        segment = Mock()
        segment.start_time = 0.0
        segment.end_time = 90.0  # 90 second segment
        segment.text = "Long segment"

        match = Mock()
        match.confidence = 0.8
        match.voiceover_segment = segment

        result = Mock()
        result.primary_match = match

        output_path = str(tmp_path / "markers.edl")
        save_timeline_as_edl([result], output_path)

        assert Path(output_path).with_suffix('.edl').exists()


class TestEdlColorMapping:
    """Tests for confidence color to EDL color mapping."""

    @pytest.fixture
    def create_match_with_confidence(self):
        """Factory to create match with specific confidence."""
        def _create(confidence):
            segment = Mock()
            segment.start_time = 0.0
            segment.end_time = 1.0
            segment.text = "Test"

            match = Mock()
            match.confidence = confidence
            match.voiceover_segment = segment

            result = Mock()
            result.primary_match = match
            return result
        return _create

    def test_confidence_green(self, tmp_path, create_match_with_confidence):
        """Test high confidence gets appropriate color."""
        from src.otio.export import save_timeline_as_edl

        result = create_match_with_confidence(0.95)
        output_path = str(tmp_path / "markers.edl")
        save_timeline_as_edl([result], output_path)

        # Should have color marker
        content = Path(output_path).with_suffix('.edl').read_text()
        assert "|C:" in content

    def test_confidence_red(self, tmp_path, create_match_with_confidence):
        """Test low confidence gets appropriate color."""
        from src.otio.export import save_timeline_as_edl

        result = create_match_with_confidence(0.15)
        output_path = str(tmp_path / "markers.edl")
        save_timeline_as_edl([result], output_path)

        content = Path(output_path).with_suffix('.edl').read_text()
        assert "|C:" in content


class TestDropFrameTimecode:
    """Tests for drop-frame timecode support in EDL export."""

    @pytest.fixture
    def mock_match_result(self):
        """Create a mock match result for testing."""
        segment = Mock()
        segment.start_time = 0.0
        segment.end_time = 5.0
        segment.text = "Test segment"

        match = Mock()
        match.confidence = 0.85
        match.voiceover_segment = segment

        result = Mock()
        result.primary_match = match
        return result

    def test_drop_frame_parameter_exists(self, tmp_path, mock_match_result):
        """Test drop_frame parameter is accepted."""
        from src.otio.export import save_timeline_as_edl

        output_path = str(tmp_path / "markers.edl")
        # Should not raise an error with drop_frame parameter
        result = save_timeline_as_edl([mock_match_result], output_path, drop_frame=True)
        assert Path(result).exists()

    def test_drop_frame_fcm_line(self, tmp_path, mock_match_result):
        """Test FCM line says 'DROP FRAME' when drop_frame=True."""
        from src.otio.export import save_timeline_as_edl

        output_path = str(tmp_path / "markers.edl")
        save_timeline_as_edl([mock_match_result], output_path, drop_frame=True)

        content = Path(output_path).with_suffix('.edl').read_text()
        assert "FCM: DROP FRAME" in content
        assert "NON-DROP FRAME" not in content

    def test_non_drop_frame_fcm_line(self, tmp_path, mock_match_result):
        """Test FCM line says 'NON-DROP FRAME' when drop_frame=False (default)."""
        from src.otio.export import save_timeline_as_edl

        output_path = str(tmp_path / "markers.edl")
        save_timeline_as_edl([mock_match_result], output_path, drop_frame=False)

        content = Path(output_path).with_suffix('.edl').read_text()
        assert "FCM: NON-DROP FRAME" in content

    def test_drop_frame_uses_semicolon_separator(self, tmp_path, mock_match_result):
        """Test drop-frame mode uses semicolons between seconds and frames."""
        from src.otio.export import save_timeline_as_edl

        output_path = str(tmp_path / "markers.edl")
        save_timeline_as_edl([mock_match_result], output_path, drop_frame=True)

        content = Path(output_path).with_suffix('.edl').read_text()
        # Drop-frame format: HH:MM:SS;FF (semicolon before frames)
        # Should have semicolon pattern like "01:00:00;00"
        import re
        pattern = r'\d{2}:\d{2}:\d{2};\d{2}'  # HH:MM:SS;FF
        matches = re.findall(pattern, content)
        assert len(matches) > 0, "Expected semicolon timecode format in drop-frame mode"

    def test_non_drop_frame_uses_colon_separator(self, tmp_path, mock_match_result):
        """Test non-drop-frame mode uses colons throughout."""
        from src.otio.export import save_timeline_as_edl

        output_path = str(tmp_path / "markers.edl")
        save_timeline_as_edl([mock_match_result], output_path, drop_frame=False)

        content = Path(output_path).with_suffix('.edl').read_text()
        # Non-drop-frame format: HH:MM:SS:FF (all colons)
        import re
        pattern = r'\d{2}:\d{2}:\d{2}:\d{2}'  # HH:MM:SS:FF
        matches = re.findall(pattern, content)
        assert len(matches) > 0, "Expected colon timecode format in non-drop-frame mode"

        # Should NOT have semicolon timecode pattern
        semicolon_pattern = r'\d{2}:\d{2}:\d{2};\d{2}'
        semicolon_matches = re.findall(semicolon_pattern, content)
        assert len(semicolon_matches) == 0, "Unexpected semicolon in non-drop-frame mode"

    def test_drop_frame_with_29_97_fps(self, tmp_path, mock_match_result):
        """Test drop-frame mode with 29.97fps (common use case)."""
        from src.otio.export import save_timeline_as_edl

        output_path = str(tmp_path / "markers.edl")
        save_timeline_as_edl([mock_match_result], output_path, frame_rate=29.97, drop_frame=True)

        content = Path(output_path).with_suffix('.edl').read_text()
        assert "FCM: DROP FRAME" in content
        # Check for semicolon format
        import re
        pattern = r'\d{2}:\d{2}:\d{2};\d{2}'
        matches = re.findall(pattern, content)
        assert len(matches) > 0

    def test_drop_frame_handles_semicolon_input_timecode(self, tmp_path, mock_match_result):
        """Test drop-frame mode handles input timecode with semicolons."""
        from src.otio.export import save_timeline_as_edl

        output_path = str(tmp_path / "markers.edl")
        # Input timecode with semicolon separator
        save_timeline_as_edl(
            [mock_match_result],
            output_path,
            timeline_start_tc="01:00:00;00",  # semicolon input
            drop_frame=True
        )

        # Should not raise an error
        assert Path(output_path).with_suffix('.edl').exists()

    def test_cmx3600_pattern_validation_drop_frame(self, tmp_path, mock_match_result):
        """Validate generated drop-frame EDL follows CMX3600 format."""
        from src.otio.export import save_timeline_as_edl
        import re

        output_path = str(tmp_path / "markers.edl")
        save_timeline_as_edl([mock_match_result], output_path, drop_frame=True)

        content = Path(output_path).with_suffix('.edl').read_text()

        # CMX3600 requires TITLE line
        assert "TITLE:" in content

        # CMX3600 requires FCM line for drop-frame
        assert "FCM: DROP FRAME" in content

        # CMX3600 event format: event_num source track edit_type timecodes
        # Example: 001  BL       V     C        01:00:00;00 01:00:00;00 01:00:00;00 01:00:00;00
        event_pattern = r'\d{3}\s+\w+\s+V\s+C\s+\d{2}:\d{2}:\d{2};\d{2}'
        event_matches = re.findall(event_pattern, content)
        assert len(event_matches) > 0, "Expected CMX3600 event format with drop-frame timecodes"

    def test_cmx3600_pattern_validation_non_drop_frame(self, tmp_path, mock_match_result):
        """Validate generated non-drop-frame EDL follows CMX3600 format."""
        from src.otio.export import save_timeline_as_edl
        import re

        output_path = str(tmp_path / "markers.edl")
        save_timeline_as_edl([mock_match_result], output_path, drop_frame=False)

        content = Path(output_path).with_suffix('.edl').read_text()

        # CMX3600 requires TITLE line
        assert "TITLE:" in content

        # CMX3600 requires FCM line for non-drop-frame
        assert "FCM: NON-DROP FRAME" in content

        # CMX3600 event format with colons for non-drop-frame
        event_pattern = r'\d{3}\s+\w+\s+V\s+C\s+\d{2}:\d{2}:\d{2}:\d{2}'
        event_matches = re.findall(event_pattern, content)
        assert len(event_matches) > 0, "Expected CMX3600 event format with non-drop-frame timecodes"


class TestReelNameGeneration:
    """Tests for reel name generation from clip paths."""

    def test_generate_reel_name_basic(self):
        """Test basic reel name generation from path."""
        from src.otio.export import _generate_reel_name

        result = _generate_reel_name("E:/videos/stock/beach_sunset.mp4")
        assert result == "STOCK_BEACH_SUNSET"

    def test_generate_reel_name_max_length(self):
        """Test reel name is truncated to max 32 chars."""
        from src.otio.export import _generate_reel_name

        # Long path that would exceed 32 chars
        long_path = "E:/very_long_folder_name/very_long_video_filename_that_exceeds_limit.mp4"
        result = _generate_reel_name(long_path)

        assert len(result) <= 32, f"Reel name exceeds 32 chars: {len(result)}"
        assert result.isupper()

    def test_generate_reel_name_sanitizes_special_chars(self):
        """Test reel name sanitizes special characters."""
        from src.otio.export import _generate_reel_name

        # Path with special chars
        result = _generate_reel_name("E:/My Videos/Beach - Sunset (HD).mp4")

        # Should only contain alphanumeric and underscore
        assert all(c.isalnum() or c == '_' for c in result)
        assert " " not in result
        assert "-" not in result
        assert "(" not in result
        assert ")" not in result

    def test_generate_reel_name_handles_unicode(self):
        """Test reel name handles unicode characters."""
        from src.otio.export import _generate_reel_name

        # Path with unicode
        result = _generate_reel_name("E:/vídeos/café_scene.mp4")

        # Should still produce valid output
        assert all(c.isalnum() or c == '_' for c in result)
        assert len(result) > 0

    def test_generate_reel_name_empty_path(self):
        """Test reel name handles empty path."""
        from src.otio.export import _generate_reel_name

        result = _generate_reel_name("")
        assert result == "BL"  # Default black

    def test_generate_reel_name_none_path(self):
        """Test reel name handles None path."""
        from src.otio.export import _generate_reel_name

        result = _generate_reel_name(None)
        assert result == "BL"  # Default black

    def test_generate_reel_name_removes_consecutive_underscores(self):
        """Test consecutive underscores are collapsed."""
        from src.otio.export import _generate_reel_name

        result = _generate_reel_name("E:/my  folder/file   name.mp4")

        # Should not have consecutive underscores
        assert "__" not in result

    def test_generate_reel_name_uppercase(self):
        """Test reel name is uppercase."""
        from src.otio.export import _generate_reel_name

        result = _generate_reel_name("E:/lowercase/filename.mp4")
        assert result == result.upper()

    def test_generate_reel_name_uses_folder_and_filename(self):
        """Test reel name includes both folder and filename."""
        from src.otio.export import _generate_reel_name

        result = _generate_reel_name("E:/project/videos/clip001.mp4")

        # Should contain parts of both folder and filename
        assert "VIDEOS" in result
        assert "CLIP" in result

    def test_generate_reel_name_custom_max_length(self):
        """Test custom max length parameter."""
        from src.otio.export import _generate_reel_name

        result = _generate_reel_name("E:/folder/filename.mp4", max_length=16)

        assert len(result) <= 16


class TestEdlWithReelNames:
    """Tests for EDL export with reel names from clip paths."""

    @pytest.fixture
    def mock_match_with_video(self):
        """Create mock match result with video segment."""
        segment = Mock()
        segment.start_time = 0.0
        segment.end_time = 5.0
        segment.text = "Test segment"

        video_segment = Mock()
        video_segment.source_file = "E:/stock/beach_sunset.mp4"
        video_segment.start_time = 10.0
        video_segment.end_time = 15.0

        match = Mock()
        match.confidence = 0.85
        match.voiceover_segment = segment
        match.video_segment = video_segment

        result = Mock()
        result.primary_match = match
        return result

    def test_edl_with_reel_names_enabled(self, tmp_path, mock_match_with_video):
        """Test EDL includes reel names when include_reel_names=True."""
        from src.otio.export import save_timeline_as_edl

        output_path = str(tmp_path / "markers.edl")
        save_timeline_as_edl([mock_match_with_video], output_path, include_reel_names=True)

        content = Path(output_path).with_suffix('.edl').read_text()

        # Should contain reel name from path (first 8 chars of generated name)
        # STOCK_BEACH_SUNSET -> first 8 = STOCK_BE
        assert "STOCK_BE" in content or "STOCK" in content

    def test_edl_without_reel_names(self, tmp_path, mock_match_with_video):
        """Test EDL uses BL when include_reel_names=False."""
        from src.otio.export import save_timeline_as_edl

        output_path = str(tmp_path / "markers.edl")
        save_timeline_as_edl([mock_match_with_video], output_path, include_reel_names=False)

        content = Path(output_path).with_suffix('.edl').read_text()

        # Should use BL as reel name
        assert "001  BL" in content

    def test_edl_reel_name_in_event_line(self, tmp_path, mock_match_with_video):
        """Test reel name appears in correct position in event line."""
        from src.otio.export import save_timeline_as_edl
        import re

        output_path = str(tmp_path / "markers.edl")
        save_timeline_as_edl([mock_match_with_video], output_path, include_reel_names=True)

        content = Path(output_path).with_suffix('.edl').read_text()

        # CMX3600 format: event_num reel_name track edit_type timecodes
        # Reel name is padded to 8 characters
        event_pattern = r'(\d{3})\s+(\w{1,8})\s+V\s+C'
        matches = re.findall(event_pattern, content)

        assert len(matches) > 0
        event_num, reel_name = matches[0]
        assert event_num == "001"
        # Reel should be derived from video path
        assert reel_name != "BL"

    def test_edl_reel_name_max_8_chars_in_event(self, tmp_path):
        """Test reel name is max 8 chars in event line (CMX3600 limit)."""
        from src.otio.export import save_timeline_as_edl
        import re

        # Create match with very long path
        segment = Mock()
        segment.start_time = 0.0
        segment.end_time = 5.0
        segment.text = "Test"

        video_segment = Mock()
        video_segment.source_file = "E:/very_long_folder_name/very_long_filename.mp4"
        video_segment.start_time = 0.0
        video_segment.end_time = 5.0

        match = Mock()
        match.confidence = 0.8
        match.voiceover_segment = segment
        match.video_segment = video_segment

        result = Mock()
        result.primary_match = match

        output_path = str(tmp_path / "markers.edl")
        save_timeline_as_edl([result], output_path, include_reel_names=True)

        content = Path(output_path).with_suffix('.edl').read_text()

        # Extract reel name from event line
        event_pattern = r'(\d{3})\s+(\S+)\s+V\s+C'
        matches = re.findall(event_pattern, content)

        assert len(matches) > 0
        _, reel_name = matches[0]
        assert len(reel_name) <= 8, f"Reel name in event line exceeds 8 chars: {reel_name}"

    def test_edl_multiple_matches_different_reels(self, tmp_path):
        """Test multiple matches get different reel names."""
        from src.otio.export import save_timeline_as_edl
        import re

        matches_list = []
        for i, video_folder in enumerate(["beach", "mountain", "city"]):
            segment = Mock()
            segment.start_time = i * 5.0
            segment.end_time = (i + 1) * 5.0
            segment.text = f"Segment {i}"

            video_segment = Mock()
            video_segment.source_file = f"E:/{video_folder}/clip{i:03d}.mp4"
            video_segment.start_time = 0.0
            video_segment.end_time = 5.0

            match = Mock()
            match.confidence = 0.8
            match.voiceover_segment = segment
            match.video_segment = video_segment

            result = Mock()
            result.primary_match = match
            matches_list.append(result)

        output_path = str(tmp_path / "markers.edl")
        save_timeline_as_edl(matches_list, output_path, include_reel_names=True)

        content = Path(output_path).with_suffix('.edl').read_text()

        # Extract all reel names
        event_pattern = r'\d{3}\s+(\S+)\s+V\s+C'
        reel_names = re.findall(event_pattern, content)

        assert len(reel_names) == 3
        # Each should be different (derived from different folders)
        unique_reels = set(reel_names)
        assert len(unique_reels) >= 2, "Expected different reel names for different source folders"


class TestTimelineSplitBySegments:
    """Tests for timeline splitting by segment count."""

    @pytest.fixture
    def timeline_with_segments(self):
        """Create a timeline with multiple segments on each track."""
        import opentimelineio as otio

        timeline = otio.schema.Timeline(name="Test Timeline")
        timeline.global_start_time = otio.opentime.RationalTime(108000, 30.0)
        timeline.metadata['Resolve_OTIO'] = {'Resolve OTIO Meta Version': '1.0'}

        # Create video tracks with 5 segments each
        for track_idx in range(3):
            track = otio.schema.Track(
                name=f"V{track_idx + 1}",
                kind=otio.schema.TrackKind.Video
            )
            for seg_idx in range(5):
                clip = otio.schema.Clip(
                    name=f"V{track_idx + 1}_Clip_{seg_idx + 1}",
                    source_range=otio.opentime.TimeRange(
                        otio.opentime.RationalTime(seg_idx * 90, 30),
                        otio.opentime.RationalTime(90, 30)
                    )
                )
                track.append(clip)
            timeline.tracks.append(track)

        # Create audio tracks with 5 segments each
        for track_idx in range(3):
            track = otio.schema.Track(
                name=f"A{track_idx + 1}",
                kind=otio.schema.TrackKind.Audio
            )
            for seg_idx in range(5):
                clip = otio.schema.Clip(
                    name=f"A{track_idx + 1}_Clip_{seg_idx + 1}",
                    source_range=otio.opentime.TimeRange(
                        otio.opentime.RationalTime(seg_idx * 90, 30),
                        otio.opentime.RationalTime(90, 30)
                    )
                )
                track.append(clip)
            timeline.tracks.append(track)

        return timeline

    def test_split_timeline_parameter_accepted(self, tmp_path, timeline_with_segments):
        """Test max_segments_per_file parameter is accepted."""
        from src.otio.export import save_timeline_split

        output_path = str(tmp_path / "output.otio")
        # Should not raise an error
        paths = save_timeline_split(timeline_with_segments, output_path, max_segments_per_file=2)
        assert len(paths) > 0

    def test_split_creates_multiple_part_files(self, tmp_path, timeline_with_segments):
        """Test splitting with max_segments_per_file=1 creates 5 part files."""
        from src.otio.export import save_timeline_split

        output_path = str(tmp_path / "output.otio")
        paths = save_timeline_split(timeline_with_segments, output_path, max_segments_per_file=1)

        # Should create part files (5 segments / 1 = 5 parts)
        part_files = [p for p in paths if '_FULL_part' in p]
        assert len(part_files) == 5

        # Verify all files exist
        for path in part_files:
            assert Path(path).exists(), f"Part file missing: {path}"

    def test_split_generates_correct_filenames(self, tmp_path, timeline_with_segments):
        """Test split generates timeline_FULL_part1.otio, timeline_FULL_part2.otio etc."""
        from src.otio.export import save_timeline_split

        output_path = str(tmp_path / "timeline.otio")
        paths = save_timeline_split(timeline_with_segments, output_path, max_segments_per_file=2)

        # With 5 segments and max=2, should get 3 parts
        part_files = [p for p in paths if '_FULL_part' in p]
        assert len(part_files) == 3

        # Check filename pattern
        expected_names = ['timeline_FULL_part1.otio', 'timeline_FULL_part2.otio', 'timeline_FULL_part3.otio']
        for expected in expected_names:
            matching = [p for p in part_files if expected in p]
            assert len(matching) == 1, f"Expected file with name containing {expected}"

    def test_split_each_part_contains_correct_segments(self, tmp_path, timeline_with_segments):
        """Test each split file contains correct segment subset."""
        import opentimelineio as otio
        from src.otio.export import save_timeline_split

        output_path = str(tmp_path / "output.otio")
        paths = save_timeline_split(timeline_with_segments, output_path, max_segments_per_file=2)

        part_files = [p for p in paths if '_FULL_part' in p]
        assert len(part_files) == 3

        # Load each part and check segment count
        part1 = otio.adapters.read_from_file(part_files[0])
        part2 = otio.adapters.read_from_file(part_files[1])
        part3 = otio.adapters.read_from_file(part_files[2])

        # Get V1 track from each part
        v1_part1 = [t for t in part1.tracks if t.name == "V1"][0]
        v1_part2 = [t for t in part2.tracks if t.name == "V1"][0]
        v1_part3 = [t for t in part3.tracks if t.name == "V1"][0]

        # Part 1 and 2 should have 2 segments each, part 3 should have 1
        assert len(list(v1_part1)) == 2, "Part 1 should have 2 segments"
        assert len(list(v1_part2)) == 2, "Part 2 should have 2 segments"
        assert len(list(v1_part3)) == 1, "Part 3 should have 1 segment"

    def test_no_split_when_segments_under_limit(self, tmp_path, timeline_with_segments):
        """Test no split occurs when segment count is under limit."""
        from src.otio.export import save_timeline_split

        output_path = str(tmp_path / "output.otio")
        # max_segments_per_file=10 but only 5 segments, so no split needed
        paths = save_timeline_split(timeline_with_segments, output_path, max_segments_per_file=10)

        # Should create single FULL file, not parts
        full_files = [p for p in paths if '_FULL.otio' in p and '_part' not in p]
        part_files = [p for p in paths if '_FULL_part' in p]

        assert len(full_files) == 1, "Should have single FULL file"
        assert len(part_files) == 0, "Should not have part files"

    def test_split_preserves_metadata(self, tmp_path, timeline_with_segments):
        """Test split files preserve Resolve_OTIO metadata."""
        import opentimelineio as otio
        from src.otio.export import save_timeline_split

        output_path = str(tmp_path / "output.otio")
        paths = save_timeline_split(timeline_with_segments, output_path, max_segments_per_file=2)

        part_files = [p for p in paths if '_FULL_part' in p]
        for part_path in part_files:
            loaded = otio.adapters.read_from_file(part_path)
            assert 'Resolve_OTIO' in loaded.metadata, f"Missing Resolve_OTIO metadata in {part_path}"

    def test_split_preserves_all_tracks(self, tmp_path, timeline_with_segments):
        """Test split files contain all tracks (video and audio)."""
        import opentimelineio as otio
        from src.otio.export import save_timeline_split

        output_path = str(tmp_path / "output.otio")
        paths = save_timeline_split(timeline_with_segments, output_path, max_segments_per_file=2)

        part_files = [p for p in paths if '_FULL_part' in p]
        for part_path in part_files:
            loaded = otio.adapters.read_from_file(part_path)
            video_tracks = [t for t in loaded.tracks if t.kind == otio.schema.TrackKind.Video]
            audio_tracks = [t for t in loaded.tracks if t.kind == otio.schema.TrackKind.Audio]

            assert len(video_tracks) == 3, f"Expected 3 video tracks in {part_path}"
            assert len(audio_tracks) == 3, f"Expected 3 audio tracks in {part_path}"

    def test_split_with_none_max_segments(self, tmp_path, timeline_with_segments):
        """Test max_segments_per_file=None behaves like no splitting."""
        from src.otio.export import save_timeline_split

        output_path = str(tmp_path / "output.otio")
        paths = save_timeline_split(timeline_with_segments, output_path, max_segments_per_file=None)

        # Should create single FULL file
        full_files = [p for p in paths if '_FULL.otio' in p and '_part' not in p]
        part_files = [p for p in paths if '_FULL_part' in p]

        assert len(full_files) == 1
        assert len(part_files) == 0

    def test_split_with_zero_max_segments(self, tmp_path, timeline_with_segments):
        """Test max_segments_per_file=0 behaves like no splitting."""
        from src.otio.export import save_timeline_split

        output_path = str(tmp_path / "output.otio")
        paths = save_timeline_split(timeline_with_segments, output_path, max_segments_per_file=0)

        # Should create single FULL file
        full_files = [p for p in paths if '_FULL.otio' in p and '_part' not in p]
        part_files = [p for p in paths if '_FULL_part' in p]

        assert len(full_files) == 1
        assert len(part_files) == 0


class TestSplitTimelineBySegmentsHelper:
    """Tests for _split_timeline_by_segments helper function."""

    def test_helper_returns_list(self):
        """Test helper function returns a list."""
        import opentimelineio as otio
        from src.otio.export import _split_timeline_by_segments

        timeline = otio.schema.Timeline(name="Test")
        track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
        for i in range(3):
            clip = otio.schema.Clip(
                name=f"Clip_{i}",
                source_range=otio.opentime.TimeRange(
                    otio.opentime.RationalTime(0, 30),
                    otio.opentime.RationalTime(90, 30)
                )
            )
            track.append(clip)
        timeline.tracks.append(track)

        result = _split_timeline_by_segments(timeline, max_segments=1)
        assert isinstance(result, list)

    def test_helper_no_video_tracks_returns_original(self):
        """Test helper returns original timeline when no video tracks."""
        import opentimelineio as otio
        from src.otio.export import _split_timeline_by_segments

        timeline = otio.schema.Timeline(name="Test")
        audio_track = otio.schema.Track(name="A1", kind=otio.schema.TrackKind.Audio)
        timeline.tracks.append(audio_track)

        result = _split_timeline_by_segments(timeline, max_segments=1)
        assert len(result) == 1
        assert result[0] is timeline

    def test_helper_segments_under_max_returns_single(self):
        """Test helper returns original when segments under max."""
        import opentimelineio as otio
        from src.otio.export import _split_timeline_by_segments

        timeline = otio.schema.Timeline(name="Test")
        track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
        clip = otio.schema.Clip(
            name="Single Clip",
            source_range=otio.opentime.TimeRange(
                otio.opentime.RationalTime(0, 30),
                otio.opentime.RationalTime(90, 30)
            )
        )
        track.append(clip)
        timeline.tracks.append(track)

        result = _split_timeline_by_segments(timeline, max_segments=10)
        assert len(result) == 1
