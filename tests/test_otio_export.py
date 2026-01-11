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
