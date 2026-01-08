"""
Unit tests for src/otio/tracks.py

Tests the track builder strategy pattern and factory.
"""

import pytest
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass
from typing import List

import opentimelineio as otio

from src.otio.tracks import (
    TrackBuilder,
    get_track_builder,
    PrimaryTrackBuilder,
    AlternativeTrackBuilder,
    DiversityTrackBuilder,
    EmbeddingDiversityTrackBuilder,
    BRollTrackBuilder,
    EntityImageTrackBuilder,
    EntityVideoTrackBuilder
)


@dataclass
class MockOutputConfig:
    """Mock output config."""
    include_alternatives: bool = True
    num_alternatives: int = 2


@dataclass
class MockConfig:
    """Mock pipeline config."""
    output: MockOutputConfig


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
class MockMatch:
    """Mock Match object."""
    file: str
    start: float
    end: float
    voiceover_segment: MockVoiceoverSegment = None

    def __post_init__(self):
        if self.voiceover_segment is None:
            self.voiceover_segment = MockVoiceoverSegment(self.start, self.end)


@dataclass
class MockMatchResult:
    """Mock MatchResult object."""
    primary: MockMatch
    primary_match: MockMatch = None

    def __post_init__(self):
        if self.primary_match is None:
            self.primary_match = self.primary


class TestTrackBuilderFactory:
    """Test get_track_builder factory function."""

    def test_factory_creates_primary_builder(self):
        """Test factory creates PrimaryTrackBuilder for track 0."""
        matches = [MockMatchResult(primary=MockMatch("test.mp4", 0.0, 5.0))]
        config = MockConfig(output=MockOutputConfig())

        builder = get_track_builder(0, matches, config, 30.0)

        assert isinstance(builder, PrimaryTrackBuilder)
        assert builder.matches == matches
        assert builder.config == config
        assert builder.frame_rate == 30.0

    def test_factory_creates_alternative_builders(self):
        """Test factory creates AlternativeTrackBuilder for tracks 1-2."""
        matches = [MockMatchResult(primary=MockMatch("test.mp4", 0.0, 5.0))]
        config = MockConfig(output=MockOutputConfig())

        builder1 = get_track_builder(1, matches, config, 30.0)
        builder2 = get_track_builder(2, matches, config, 30.0)

        assert isinstance(builder1, AlternativeTrackBuilder)
        assert isinstance(builder2, AlternativeTrackBuilder)

    def test_factory_creates_diversity_builders(self):
        """Test factory creates DiversityTrackBuilder for tracks 3-5."""
        matches = [MockMatchResult(primary=MockMatch("test.mp4", 0.0, 5.0))]
        config = MockConfig(output=MockOutputConfig())

        builder3 = get_track_builder(3, matches, config, 30.0)
        builder4 = get_track_builder(4, matches, config, 30.0)
        builder5 = get_track_builder(5, matches, config, 30.0)

        assert isinstance(builder3, DiversityTrackBuilder)
        assert isinstance(builder4, DiversityTrackBuilder)
        assert isinstance(builder5, DiversityTrackBuilder)

    def test_factory_creates_embedding_diversity_builder(self):
        """Test factory creates EmbeddingDiversityTrackBuilder for track 6."""
        matches = [MockMatchResult(primary=MockMatch("test.mp4", 0.0, 5.0))]
        config = MockConfig(output=MockOutputConfig())

        builder = get_track_builder(6, matches, config, 30.0)

        assert isinstance(builder, EmbeddingDiversityTrackBuilder)

    def test_factory_creates_broll_builder(self):
        """Test factory creates BRollTrackBuilder for track 7."""
        matches = [MockMatchResult(primary=MockMatch("test.mp4", 0.0, 5.0))]
        config = MockConfig(output=MockOutputConfig())

        builder = get_track_builder(7, matches, config, 30.0)

        assert isinstance(builder, BRollTrackBuilder)

    def test_factory_creates_entity_image_builder(self):
        """Test factory creates EntityImageTrackBuilder for track 8."""
        matches = [MockMatchResult(primary=MockMatch("test.mp4", 0.0, 5.0))]
        config = MockConfig(output=MockOutputConfig())

        builder = get_track_builder(8, matches, config, 30.0)

        assert isinstance(builder, EntityImageTrackBuilder)

    def test_factory_creates_entity_video_builder(self):
        """Test factory creates EntityVideoTrackBuilder for track 9."""
        matches = [MockMatchResult(primary=MockMatch("test.mp4", 0.0, 5.0))]
        config = MockConfig(output=MockOutputConfig())

        builder = get_track_builder(9, matches, config, 30.0)

        assert isinstance(builder, EntityVideoTrackBuilder)

    def test_factory_raises_error_for_invalid_track(self):
        """Test factory raises ValueError for invalid track index."""
        matches = [MockMatchResult(primary=MockMatch("test.mp4", 0.0, 5.0))]
        config = MockConfig(output=MockOutputConfig())

        with pytest.raises(ValueError, match="No builder for track index"):
            get_track_builder(99, matches, config, 30.0)

    def test_factory_passes_kwargs_to_builder(self):
        """Test factory passes kwargs to builder."""
        matches = [MockMatchResult(primary=MockMatch("test.mp4", 0.0, 5.0))]
        config = MockConfig(output=MockOutputConfig())

        builder = get_track_builder(
            0, matches, config, 30.0,
            entity_images={"test": "data"},
            segment_lookup={"video1": []}
        )

        assert "entity_images" in builder.kwargs
        assert builder.kwargs["entity_images"] == {"test": "data"}
        assert "segment_lookup" in builder.kwargs


class TestTrackNames:
    """Test _get_track_name method."""

    def test_track_names_are_correct(self):
        """Test that track names match expected values."""
        matches = [MockMatchResult(primary=MockMatch("test.mp4", 0.0, 5.0))]
        config = MockConfig(output=MockOutputConfig())

        expected_names = [
            "Primary Video",
            "Alternative Video 1",
            "Alternative Video 2",
            "Secondary Diversity 1",
            "Secondary Diversity 2",
            "Secondary Diversity 3",
            "Embedding-Diversity Strategy",
            "B-roll Only",
            "Entity Images (Google)",
            "Stock Videos (Pexels/Pixabay)",
        ]

        for idx, expected in enumerate(expected_names):
            builder = get_track_builder(idx, matches, config, 30.0)
            name = builder._get_track_name(idx)
            assert name == expected

    def test_track_name_fallback_for_high_indices(self):
        """Test that track names fallback to generic name for high indices."""
        matches = [MockMatchResult(primary=MockMatch("test.mp4", 0.0, 5.0))]
        config = MockConfig(output=MockOutputConfig())
        builder = PrimaryTrackBuilder(matches, config, 30.0)

        # Test fallback for index beyond defined names
        name = builder._get_track_name(99)
        assert name == "Track 100"  # idx + 1


class TestPrimaryTrackBuilder:
    """Test PrimaryTrackBuilder class."""

    def test_primary_builder_creates_tracks(self):
        """Test that PrimaryTrackBuilder creates video and audio tracks."""
        matches = [MockMatchResult(primary=MockMatch("test.mp4", 0.0, 5.0))]
        config = MockConfig(output=MockOutputConfig())

        builder = PrimaryTrackBuilder(matches, config, 30.0)
        video_track, audio_track = builder.build(0)

        assert isinstance(video_track, otio.schema.Track)
        assert isinstance(audio_track, otio.schema.Track)
        assert video_track.kind == otio.schema.TrackKind.Video
        assert audio_track.kind == otio.schema.TrackKind.Audio

    def test_primary_tracks_are_enabled(self):
        """Test that primary tracks are enabled by default."""
        matches = [MockMatchResult(primary=MockMatch("test.mp4", 0.0, 5.0))]
        config = MockConfig(output=MockOutputConfig())

        builder = PrimaryTrackBuilder(matches, config, 30.0)
        video_track, audio_track = builder.build(0)

        assert video_track.enabled is True
        assert audio_track.enabled is True

    def test_primary_track_name(self):
        """Test that primary track has correct name."""
        matches = [MockMatchResult(primary=MockMatch("test.mp4", 0.0, 5.0))]
        config = MockConfig(output=MockOutputConfig())

        builder = PrimaryTrackBuilder(matches, config, 30.0)
        video_track, audio_track = builder.build(0)

        assert "Primary Video" in video_track.name


class TestAlternativeTrackBuilder:
    """Test AlternativeTrackBuilder class."""

    def test_alternative_tracks_are_disabled(self):
        """Test that alternative tracks are disabled by default."""
        matches = [MockMatchResult(primary=MockMatch("test.mp4", 0.0, 5.0))]
        config = MockConfig(output=MockOutputConfig())

        builder = AlternativeTrackBuilder(matches, config, 30.0)
        video_track, audio_track = builder.build(1)

        assert video_track.enabled is False
        assert audio_track.enabled is False

    def test_alternative_track_names(self):
        """Test that alternative tracks have correct names."""
        matches = [MockMatchResult(primary=MockMatch("test.mp4", 0.0, 5.0))]
        config = MockConfig(output=MockOutputConfig())

        builder1 = AlternativeTrackBuilder(matches, config, 30.0)
        video_track1, _ = builder1.build(1)

        builder2 = AlternativeTrackBuilder(matches, config, 30.0)
        video_track2, _ = builder2.build(2)

        assert "Alternative Video 1" in video_track1.name
        assert "Alternative Video 2" in video_track2.name


class TestDiversityTrackBuilder:
    """Test DiversityTrackBuilder class."""

    def test_diversity_tracks_are_disabled(self):
        """Test that diversity tracks are disabled by default."""
        matches = [MockMatchResult(primary=MockMatch("test.mp4", 0.0, 5.0))]
        config = MockConfig(output=MockOutputConfig())

        builder = DiversityTrackBuilder(matches, config, 30.0)
        video_track, audio_track = builder.build(3)

        assert video_track.enabled is False
        assert audio_track.enabled is False


class TestStrategyTrackBuilders:
    """Test strategy-specific track builders."""

    def test_embedding_diversity_track_disabled(self):
        """Test that embedding diversity track is disabled by default."""
        matches = [MockMatchResult(primary=MockMatch("test.mp4", 0.0, 5.0))]
        config = MockConfig(output=MockOutputConfig())

        builder = EmbeddingDiversityTrackBuilder(matches, config, 30.0)
        video_track, audio_track = builder.build(6)

        assert video_track.enabled is False
        assert audio_track.enabled is False

    def test_broll_track_disabled(self):
        """Test that B-roll track is disabled by default."""
        matches = [MockMatchResult(primary=MockMatch("test.mp4", 0.0, 5.0))]
        config = MockConfig(output=MockOutputConfig())

        builder = BRollTrackBuilder(matches, config, 30.0)
        video_track, audio_track = builder.build(7)

        assert video_track.enabled is False
        assert audio_track.enabled is False


class TestEntityTrackBuilders:
    """Test entity image and video track builders."""

    def test_entity_image_track_without_images(self):
        """Test EntityImageTrackBuilder without entity images."""
        matches = [MockMatchResult(primary=MockMatch("test.mp4", 0.0, 5.0))]
        config = MockConfig(output=MockOutputConfig())

        builder = EntityImageTrackBuilder(matches, config, 30.0)
        video_track, audio_track = builder.build(8)

        assert isinstance(video_track, otio.schema.Track)
        assert video_track.enabled is False

    # NOTE: Testing with entity images requires complex config mocking (image_search attribute)
    # Entity image integration is tested in test_otio_integration.py

    def test_entity_video_track_without_videos(self):
        """Test EntityVideoTrackBuilder without entity videos."""
        matches = [MockMatchResult(primary=MockMatch("test.mp4", 0.0, 5.0))]
        config = MockConfig(output=MockOutputConfig())

        builder = EntityVideoTrackBuilder(matches, config, 30.0)
        video_track, audio_track = builder.build(9)

        assert isinstance(video_track, otio.schema.Track)
        assert video_track.enabled is False

    # NOTE: Testing with entity videos requires complex config mocking (image_search attribute)
    # Entity video integration is tested in test_otio_integration.py


class TestTrackBuilderAbstract:
    """Test TrackBuilder abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that TrackBuilder cannot be instantiated directly."""
        matches = [MockMatchResult(primary=MockMatch("test.mp4", 0.0, 5.0))]
        config = MockConfig(output=MockOutputConfig())

        # This should raise TypeError because build() is abstract
        with pytest.raises(TypeError):
            TrackBuilder(matches, config, 30.0)

    def test_subclass_must_implement_build(self):
        """Test that subclass must implement build method."""
        class IncompleteBuilder(TrackBuilder):
            pass  # Missing build() implementation

        matches = [MockMatchResult(primary=MockMatch("test.mp4", 0.0, 5.0))]
        config = MockConfig(output=MockOutputConfig())

        with pytest.raises(TypeError):
            IncompleteBuilder(matches, config, 30.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
