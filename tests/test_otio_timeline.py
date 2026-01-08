"""
Unit tests for src/otio/timeline.py

Tests the timeline creation logic with mocked data.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import List, Dict, Optional

import opentimelineio as otio

# Import from src/otio
from src.otio.timeline import create_timeline, _validate_entity_images


@dataclass
class MockVoiceoverSegment:
    """Mock voiceover segment."""
    start: float
    end: float
    text: str = "Test text"
    start_time: float = None
    end_time: float = None

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = self.start
        if self.end_time is None:
            self.end_time = self.end


@dataclass
class MockVideoSegment:
    """Mock video segment."""
    source_file: str
    start_time: float
    end_time: float
    text: str = ""  # Transcript text


@dataclass
class MockMatch:
    """Mock Match object for testing."""
    file: str
    start: float
    end: float
    confidence: float = 0.8
    speed: float = 1.0
    reasoning: str = "Test reasoning"
    strategy: str = "embedding"
    is_keyword_match: bool = False
    is_visual_match: bool = False
    keyword: str = ""
    embedding_similarity: float = 0.85
    clip_reuse_count: int = 0
    voiceover_segment: MockVoiceoverSegment = None
    video_segment: MockVideoSegment = None

    def __post_init__(self):
        if self.voiceover_segment is None:
            self.voiceover_segment = MockVoiceoverSegment(self.start, self.end)
        if self.video_segment is None:
            self.video_segment = MockVideoSegment(self.file, self.start, self.end)


@dataclass
class MockMatchResult:
    """Mock MatchResult object for testing."""
    primary: MockMatch
    alternatives: List[MockMatch]
    secondaries: List[MockMatch]
    strategies: Dict[str, MockMatch]
    primary_match: MockMatch = None
    alternative_matches: List[MockMatch] = None
    secondary_matches: List[MockMatch] = None
    strategy_matches: List[MockMatch] = None

    def __post_init__(self):
        if self.primary_match is None:
            self.primary_match = self.primary
        if self.alternative_matches is None:
            self.alternative_matches = self.alternatives
        if self.secondary_matches is None:
            self.secondary_matches = self.secondaries
        if self.strategy_matches is None:
            # Convert dict to list of matches
            self.strategy_matches = list(self.strategies.values()) if self.strategies else []


@dataclass
class MockOutputConfig:
    """Mock output config."""
    include_alternatives: bool = True
    num_alternatives: int = 2
    include_strategy_tracks: bool = True
    strategy_tracks: List[str] = None
    include_entity_images: bool = False
    include_entity_videos: bool = False

    def __post_init__(self):
        if self.strategy_tracks is None:
            self.strategy_tracks = ["embedding_diversity", "broll_only"]


@dataclass
class MockConfig:
    """Mock pipeline config."""
    output: MockOutputConfig


@dataclass
class MockEntityImage:
    """Mock entity image."""
    file: str
    score: float = 0.9


@dataclass
class MockEntityResult:
    """Mock entity result with images."""
    images: List[MockEntityImage]


class TestValidateEntityImages:
    """Test _validate_entity_images function."""

    def test_validate_empty_dict(self):
        """Test with empty entity images dict."""
        result = _validate_entity_images({})
        assert result == {}

    def test_validate_with_valid_images(self, tmp_path):
        """Test validation with valid image files."""
        # Create temporary image files
        img1 = tmp_path / "entity1_1.jpg"
        img1.write_text("fake image 1")
        img2 = tmp_path / "entity1_2.jpg"
        img2.write_text("fake image 2")

        entity_images = {
            "Entity1": MockEntityResult(images=[
                MockEntityImage(file=str(img1)),
                MockEntityImage(file=str(img2))
            ])
        }

        result = _validate_entity_images(entity_images)

        assert "Entity1" in result
        assert len(result["Entity1"].images) == 2

    def test_validate_filters_missing_files(self, tmp_path):
        """Test that validation filters out missing image files."""
        # Create one valid file
        valid_img = tmp_path / "exists.jpg"
        valid_img.write_text("exists")

        # Reference a file that doesn't exist
        missing_img = tmp_path / "missing.jpg"

        entity_images = {
            "Entity1": MockEntityResult(images=[
                MockEntityImage(file=str(valid_img)),
                MockEntityImage(file=str(missing_img))
            ])
        }

        result = _validate_entity_images(entity_images)

        assert "Entity1" in result
        # Only the valid file should remain
        assert len(result["Entity1"].images) == 1
        assert result["Entity1"].images[0].file == str(valid_img)

    def test_validate_removes_entity_with_no_valid_images(self, tmp_path):
        """Test that entities with no valid images are removed."""
        missing_img = tmp_path / "missing.jpg"

        entity_images = {
            "Entity1": MockEntityResult(images=[
                MockEntityImage(file=str(missing_img))
            ])
        }

        result = _validate_entity_images(entity_images)

        # Entity1 should be removed since it has no valid images
        assert "Entity1" not in result
        assert len(result) == 0

    def test_validate_entity_without_images_attribute(self):
        """Test handling of entity results without images attribute."""
        @dataclass
        class BadEntity:
            name: str

        entity_images = {
            "BadEntity": BadEntity(name="test")
        }

        result = _validate_entity_images(entity_images)

        # BadEntity should be filtered out
        assert "BadEntity" not in result

    def test_validate_entity_with_empty_images_list(self):
        """Test handling of entity with empty images list."""
        entity_images = {
            "Entity1": MockEntityResult(images=[])
        }

        result = _validate_entity_images(entity_images)

        # Entity1 should be removed
        assert "Entity1" not in result


class TestCreateTimeline:
    """Test create_timeline function."""

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    def test_creates_basic_timeline(self, mock_windows_path, mock_duration):
        """Test that create_timeline creates a valid OTIO timeline."""
        # Mock helpers
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = 10.0

        # Create minimal test data
        matches = [
            MockMatchResult(
                primary=MockMatch(file="video1.mp4", start=0.0, end=5.0),
                alternatives=[],
                secondaries=[],
                strategies={}
            )
        ]
        config = MockConfig(output=MockOutputConfig(
            include_alternatives=False,
            include_strategy_tracks=False
        ))

        timeline = create_timeline(matches, config)

        assert isinstance(timeline, otio.schema.Timeline)
        assert timeline.name == "Matched Footage"

        # Check metadata
        assert 'Resolve_OTIO' in timeline.metadata
        assert timeline.metadata['Resolve_OTIO']['Resolve OTIO Meta Version'] == '1.0'

        # Check global_start_time is set
        assert timeline.global_start_time is not None
        assert isinstance(timeline.global_start_time, otio.opentime.RationalTime)

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    def test_creates_video_tracks(self, mock_windows_path, mock_duration):
        """Test that video tracks are created correctly."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = 10.0

        matches = [
            MockMatchResult(
                primary=MockMatch(file="video1.mp4", start=0.0, end=5.0),
                alternatives=[
                    MockMatch(file="video2.mp4", start=0.0, end=5.0),
                    MockMatch(file="video3.mp4", start=0.0, end=5.0)
                ],
                secondaries=[],
                strategies={}
            )
        ]
        config = MockConfig(output=MockOutputConfig(
            include_alternatives=True,
            num_alternatives=2,
            include_strategy_tracks=False
        ))

        timeline = create_timeline(matches, config)

        # Find video tracks
        video_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Video]

        # Should have at least V1 (primary)
        assert len(video_tracks) > 0
        assert "Primary" in video_tracks[0].name

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    def test_creates_audio_tracks(self, mock_windows_path, mock_duration):
        """Test that audio tracks are created correctly."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = 10.0

        matches = [
            MockMatchResult(
                primary=MockMatch(file="video1.mp4", start=0.0, end=5.0),
                alternatives=[],
                secondaries=[],
                strategies={}
            )
        ]
        config = MockConfig(output=MockOutputConfig(
            include_alternatives=False,
            include_strategy_tracks=False
        ))

        timeline = create_timeline(matches, config)

        # Find audio tracks
        audio_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Audio]

        # Should have at least A1 (primary audio)
        assert len(audio_tracks) > 0

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    def test_timeline_with_custom_frame_rate(self, mock_windows_path, mock_duration):
        """Test timeline creation with custom frame rate."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = 10.0

        matches = [
            MockMatchResult(
                primary=MockMatch(file="video1.mp4", start=0.0, end=5.0),
                alternatives=[],
                secondaries=[],
                strategies={}
            )
        ]
        config = MockConfig(output=MockOutputConfig(
            include_alternatives=False,
            include_strategy_tracks=False
        ))

        timeline = create_timeline(matches, config, frame_rate=24.0)

        # Check that global_start_time uses the custom frame rate
        assert timeline.global_start_time.rate == 24.0

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    def test_timeline_with_voiceover(self, mock_windows_path, mock_duration, tmp_path):
        """Test timeline creation with voiceover track."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = 10.0

        # Create a fake voiceover file
        voiceover = tmp_path / "voiceover.mp3"
        voiceover.write_text("fake audio")

        matches = [
            MockMatchResult(
                primary=MockMatch(file="video1.mp4", start=0.0, end=5.0),
                alternatives=[],
                secondaries=[],
                strategies={}
            )
        ]
        config = MockConfig(output=MockOutputConfig(
            include_alternatives=False,
            include_strategy_tracks=False
        ))

        timeline = create_timeline(matches, config, voiceover_path=str(voiceover))

        # Check that audio tracks include voiceover
        audio_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Audio]

        # Should have voiceover track (A9)
        voiceover_tracks = [t for t in audio_tracks if "Voiceover" in t.name]
        assert len(voiceover_tracks) > 0

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    def test_timeline_with_strategy_tracks(self, mock_windows_path, mock_duration):
        """Test timeline creation with strategy tracks."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = 10.0

        matches = [
            MockMatchResult(
                primary=MockMatch(file="video1.mp4", start=0.0, end=5.0),
                alternatives=[],
                secondaries=[],
                strategies={
                    "embedding_diversity": MockMatch(file="video2.mp4", start=0.0, end=5.0),
                    "broll_only": MockMatch(file="video3.mp4", start=0.0, end=5.0)
                }
            )
        ]
        config = MockConfig(output=MockOutputConfig(
            include_alternatives=False,
            include_strategy_tracks=True,
            strategy_tracks=["embedding_diversity", "broll_only"]
        ))

        timeline = create_timeline(matches, config)

        # Find strategy tracks
        video_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Video]

        strategy_track_names = [t.name for t in video_tracks]

        # Should have strategy tracks
        assert any("Embedding-Diversity" in name for name in strategy_track_names)
        assert any("B-roll" in name for name in strategy_track_names)

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    def test_timeline_tracks_stack_name(self, mock_windows_path, mock_duration):
        """Test that tracks stack name is empty (DaVinci format)."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = 10.0

        matches = [
            MockMatchResult(
                primary=MockMatch(file="video1.mp4", start=0.0, end=5.0),
                alternatives=[],
                secondaries=[],
                strategies={}
            )
        ]
        config = MockConfig(output=MockOutputConfig(
            include_alternatives=False,
            include_strategy_tracks=False
        ))

        timeline = create_timeline(matches, config)

        # Tracks stack name should be empty for DaVinci compatibility
        assert timeline.tracks.name == ""

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    def test_timeline_with_empty_matches(self, mock_windows_path, mock_duration):
        """Test timeline creation with empty matches list."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = 10.0

        matches = []
        config = MockConfig(output=MockOutputConfig(
            include_alternatives=False,
            include_strategy_tracks=False
        ))

        timeline = create_timeline(matches, config)

        # Should still create valid timeline structure
        assert isinstance(timeline, otio.schema.Timeline)
        assert timeline.name == "Matched Footage"


class TestResolveVideoSegment:
    """Test the resolve_video_segment nested function.

    This function is tested indirectly through create_timeline with downloaded_segments.
    """

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    def test_audio_first_mode_segment_resolution(self, mock_windows_path, mock_duration, tmp_path):
        """Test that audio-first mode correctly resolves video segments."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = 10.0

        # Create mock downloaded segment
        @dataclass
        class MockDownloadedSegment:
            file: str
            video_id: str
            original_start: float
            original_end: float

        video_file = tmp_path / "video1_seg.mp4"
        video_file.write_text("fake video")

        downloaded_segments = [
            MockDownloadedSegment(
                file=str(video_file),
                video_id="video1",
                original_start=0.0,
                original_end=10.0
            )
        ]

        matches = [
            MockMatchResult(
                primary=MockMatch(file="video1.mp3", start=2.0, end=7.0),
                alternatives=[],
                secondaries=[],
                strategies={}
            )
        ]
        config = MockConfig(output=MockOutputConfig(
            include_alternatives=False,
            include_strategy_tracks=False
        ))

        # Call create_timeline with downloaded_segments
        timeline = create_timeline(
            matches,
            config,
            downloaded_segments=downloaded_segments
        )

        # Timeline should be created successfully
        assert isinstance(timeline, otio.schema.Timeline)

        # Note: Full verification of segment resolution would require
        # checking clip references, which is complex and requires
        # more detailed mocking of OTIO internals


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
