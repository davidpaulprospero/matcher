"""
Unit tests for src/otio/tracks.py

Tests the track builder strategy pattern and factory.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass
from typing import List

import opentimelineio as otio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
class MockImageSearchConfig:
    """Mock image search config."""
    enable_sticky_matching: bool = False
    provider: str = "google"


@dataclass
class MockConfig:
    """Mock pipeline config."""
    output: MockOutputConfig
    image_search: MockImageSearchConfig = None

    def __post_init__(self):
        if self.image_search is None:
            self.image_search = MockImageSearchConfig()


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
    text: str = "Test video text"


@dataclass
class MockMatch:
    """Mock Match object."""
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
    """Mock MatchResult object."""
    primary: MockMatch
    alternatives: List[MockMatch] = None
    secondary_matches: List[MockMatch] = None
    strategy_matches: List[MockMatch] = None
    primary_match: MockMatch = None

    def __post_init__(self):
        if self.primary_match is None:
            self.primary_match = self.primary
        if self.alternatives is None:
            self.alternatives = []
        if self.secondary_matches is None:
            self.secondary_matches = []
        if self.strategy_matches is None:
            self.strategy_matches = []


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


class TestHelperMethods:
    """Test TrackBuilder helper methods."""

    def test_create_clip_basic(self):
        """Test _create_clip with basic match."""
        match = MockMatch("C:/Videos/test.mp4", 1.0, 6.0)
        matches = [MockMatchResult(primary=match)]
        config = MockConfig(output=MockOutputConfig())
        builder = PrimaryTrackBuilder(matches, config, 30.0)

        clip = builder._create_clip(
            match=match,
            segment_id="S001",
            clip_label="PRIMARY",
            target_duration=5.0
        )

        assert isinstance(clip, otio.schema.Clip)
        assert "S001" in clip.name
        assert "PRIMARY" in clip.name

    def test_create_clip_with_metadata(self):
        """Test _create_clip with custom metadata."""
        match = MockMatch("C:/Videos/test.mp4", 0.0, 5.0, confidence=0.95)
        matches = [MockMatchResult(primary=match)]
        config = MockConfig(output=MockOutputConfig())
        builder = PrimaryTrackBuilder(matches, config, 30.0)

        metadata = {"test_key": "test_value", "confidence": 0.95}
        clip = builder._create_clip(
            match=match,
            segment_id="S000",
            clip_label="TEST",
            target_duration=5.0,
            metadata=metadata
        )

        assert clip.metadata["test_key"] == "test_value"
        assert clip.metadata["confidence"] == 0.95

    def test_create_clip_with_resolve_function(self):
        """Test _create_clip with audio-first resolution."""
        match = MockMatch("audio.mp3", 0.0, 5.0)
        matches = [MockMatchResult(primary=match)]
        config = MockConfig(output=MockOutputConfig())

        # Mock resolve function
        def mock_resolve(source_file, start_time):
            return "video.mp4", start_time + 1.0

        builder = PrimaryTrackBuilder(
            matches, config, 30.0,
            resolve_video_segment=mock_resolve
        )

        clip = builder._create_clip(
            match=match,
            segment_id="S000",
            clip_label="RESOLVED",
            target_duration=5.0
        )

        # Clip should reference resolved video file
        assert "video.mp4" in clip.media_reference.target_url or "video" in clip.name

    def test_create_gap(self):
        """Test _create_gap creates proper gap clip."""
        matches = [MockMatchResult(primary=MockMatch("test.mp4", 0.0, 5.0))]
        config = MockConfig(output=MockOutputConfig())
        builder = PrimaryTrackBuilder(matches, config, 30.0)

        gap = builder._create_gap(150)  # 5 seconds at 30fps

        assert isinstance(gap, otio.schema.Gap)
        assert gap.source_range.duration.value == 150
        assert gap.source_range.duration.rate == 30.0

    def test_create_clip_with_segment_offset(self):
        """Test _create_clip with segment file offset."""
        # Simulate segment file name pattern: video_0120.mp4 (offset = 120.0s)
        match = MockMatch("C:/Videos/video_0120.mp4", 5.0, 10.0)
        matches = [MockMatchResult(primary=match)]
        config = MockConfig(output=MockOutputConfig())
        builder = PrimaryTrackBuilder(matches, config, 30.0)

        clip = builder._create_clip(
            match=match,
            segment_id="S000",
            clip_label="SEGMENT",
            target_duration=5.0
        )

        # Segment offset logic adjusts start time
        assert isinstance(clip, otio.schema.Clip)
        assert "S000" in clip.name


class TestAlternativeBuilderWithMatches:
    """Test AlternativeTrackBuilder with actual alternative matches."""

    def test_alternative_builder_with_alternatives(self):
        """Test alternative track with alternatives present."""
        primary = MockMatch("primary.mp4", 0.0, 5.0, confidence=0.9)
        alt1 = MockMatch("alt1.mp4", 0.0, 5.0, confidence=0.8)
        alt2 = MockMatch("alt2.mp4", 0.0, 5.0, confidence=0.7)

        matches = [MockMatchResult(
            primary=primary,
            alternatives=[alt1, alt2]
        )]
        config = MockConfig(output=MockOutputConfig())

        builder = AlternativeTrackBuilder(matches, config, 30.0)
        video_track, audio_track = builder.build(1)

        # Should have 1 clip (alt1) in V2
        assert len(video_track) == 1
        assert len(audio_track) == 1
        assert isinstance(video_track[0], otio.schema.Clip)
        assert "ALT1" in video_track[0].name

    def test_alternative_builder_without_alternatives(self):
        """Test alternative track without alternatives (creates gaps)."""
        primary = MockMatch("primary.mp4", 0.0, 5.0, confidence=0.9)

        matches = [MockMatchResult(primary=primary, alternatives=[])]
        config = MockConfig(output=MockOutputConfig())

        builder = AlternativeTrackBuilder(matches, config, 30.0)
        video_track, audio_track = builder.build(1)

        # Should have 1 gap
        assert len(video_track) == 1
        assert isinstance(video_track[0], otio.schema.Gap)


class TestDiversityBuilderWithMatches:
    """Test DiversityTrackBuilder with actual secondary matches."""

    def test_diversity_builder_with_secondaries(self):
        """Test diversity track with secondary matches."""
        primary = MockMatch("primary.mp4", 0.0, 5.0, confidence=0.9)
        sec1 = MockMatch("sec1.mp4", 0.0, 5.0, confidence=0.75)
        sec2 = MockMatch("sec2.mp4", 0.0, 5.0, confidence=0.7)
        sec3 = MockMatch("sec3.mp4", 0.0, 5.0, confidence=0.65)

        matches = [MockMatchResult(
            primary=primary,
            secondary_matches=[sec1, sec2, sec3]
        )]
        config = MockConfig(output=MockOutputConfig())

        builder = DiversityTrackBuilder(matches, config, 30.0)
        video_track, audio_track = builder.build(3)  # V4 = first secondary

        # Should have 1 clip (sec1) in V4
        assert len(video_track) == 1
        assert isinstance(video_track[0], otio.schema.Clip)
        assert "Secondary" in video_track[0].name

    def test_diversity_builder_without_secondaries(self):
        """Test diversity track without secondary matches (creates gaps)."""
        primary = MockMatch("primary.mp4", 0.0, 5.0, confidence=0.9)

        matches = [MockMatchResult(primary=primary, secondary_matches=[])]
        config = MockConfig(output=MockOutputConfig())

        builder = DiversityTrackBuilder(matches, config, 30.0)
        video_track, audio_track = builder.build(3)

        # Should have 1 gap
        assert len(video_track) == 1
        assert isinstance(video_track[0], otio.schema.Gap)


class TestStrategyBuildersWithMatches:
    """Test strategy builders with actual strategy matches."""

    def test_embedding_diversity_with_strategy_match(self):
        """Test embedding diversity track with strategy match."""
        primary = MockMatch("primary.mp4", 0.0, 5.0, confidence=0.9)
        emb_div = MockMatch("embdiv.mp4", 0.0, 5.0, confidence=0.8, strategy="embedding_diversity")

        matches = [MockMatchResult(
            primary=primary,
            strategy_matches=[emb_div]
        )]
        config = MockConfig(output=MockOutputConfig())

        builder = EmbeddingDiversityTrackBuilder(matches, config, 30.0)
        video_track, audio_track = builder.build(6)  # V7

        # Should have 1 clip
        assert len(video_track) == 1
        assert isinstance(video_track[0], otio.schema.Clip)
        assert video_track[0].metadata.get("strategy") == "embedding_diversity"

    def test_embedding_diversity_without_strategy_match(self):
        """Test embedding diversity track without strategy match (creates gap)."""
        primary = MockMatch("primary.mp4", 0.0, 5.0, confidence=0.9)

        matches = [MockMatchResult(primary=primary, strategy_matches=[])]
        config = MockConfig(output=MockOutputConfig())

        builder = EmbeddingDiversityTrackBuilder(matches, config, 30.0)
        video_track, audio_track = builder.build(6)

        # Should have 1 gap
        assert len(video_track) == 1
        assert isinstance(video_track[0], otio.schema.Gap)

    def test_broll_with_strategy_match(self):
        """Test B-roll track with strategy match."""
        primary = MockMatch("primary.mp4", 0.0, 5.0, confidence=0.9)
        broll = MockMatch("broll.mp4", 0.0, 5.0, confidence=0.75, strategy="broll_only")

        matches = [MockMatchResult(
            primary=primary,
            strategy_matches=[broll]
        )]
        config = MockConfig(output=MockOutputConfig())

        builder = BRollTrackBuilder(matches, config, 30.0)
        video_track, audio_track = builder.build(7)  # V8

        # Should have 1 clip
        assert len(video_track) == 1
        assert isinstance(video_track[0], otio.schema.Clip)
        assert video_track[0].metadata.get("strategy") == "broll_only"

    def test_broll_without_strategy_match(self):
        """Test B-roll track without strategy match (creates gap)."""
        primary = MockMatch("primary.mp4", 0.0, 5.0, confidence=0.9)

        matches = [MockMatchResult(primary=primary, strategy_matches=[])]
        config = MockConfig(output=MockOutputConfig())

        builder = BRollTrackBuilder(matches, config, 30.0)
        video_track, audio_track = builder.build(7)

        # Should have 1 gap
        assert len(video_track) == 1
        assert isinstance(video_track[0], otio.schema.Gap)

    def test_mixed_strategies_in_strategy_matches(self):
        """Test finding correct strategy when multiple strategies present."""
        primary = MockMatch("primary.mp4", 0.0, 5.0, confidence=0.9)
        emb_div = MockMatch("embdiv.mp4", 0.0, 5.0, confidence=0.8, strategy="embedding_diversity")
        broll = MockMatch("broll.mp4", 0.0, 5.0, confidence=0.75, strategy="broll_only")

        matches = [MockMatchResult(
            primary=primary,
            strategy_matches=[emb_div, broll]
        )]
        config = MockConfig(output=MockOutputConfig())

        # Test embedding diversity builder finds its match
        emb_builder = EmbeddingDiversityTrackBuilder(matches, config, 30.0)
        emb_video, _ = emb_builder.build(6)
        assert isinstance(emb_video[0], otio.schema.Clip)
        assert "embdiv" in emb_video[0].media_reference.target_url.lower()

        # Test B-roll builder finds its match
        broll_builder = BRollTrackBuilder(matches, config, 30.0)
        broll_video, _ = broll_builder.build(7)
        assert isinstance(broll_video[0], otio.schema.Clip)
        assert "broll" in broll_video[0].media_reference.target_url.lower()


class TestPrimaryBuilderClipCreation:
    """Test PrimaryTrackBuilder clip creation logic."""

    def test_primary_builder_creates_clips_with_metadata(self):
        """Test that primary builder creates clips with all metadata fields."""
        match = MockMatch(
            "C:/Videos/test.mp4", 0.0, 5.0,
            confidence=0.9,
            reasoning="Test reasoning",
            is_keyword_match=True,
            is_visual_match=False,
            embedding_similarity=0.85,
            clip_reuse_count=2
        )
        match.voiceover_segment = MockVoiceoverSegment(0.0, 5.0, "Test voiceover text")
        match.video_segment = MockVideoSegment("C:/Videos/test.mp4", 0.0, 5.0, "Test video text")

        matches = [MockMatchResult(primary=match)]
        config = MockConfig(output=MockOutputConfig())

        builder = PrimaryTrackBuilder(matches, config, 30.0)
        video_track, audio_track = builder.build(0)

        clip = video_track[0]
        assert clip.metadata["confidence"] == 0.9
        assert clip.metadata["reasoning"] == "Test reasoning"
        assert clip.metadata["is_keyword_match"] is True
        assert clip.metadata["is_visual_match"] is False
        assert clip.metadata["embedding_similarity"] == 0.85
        assert clip.metadata["reuse_count"] == 2
        assert "voiceover_text" in clip.metadata
        assert "video_text" in clip.metadata

    def test_primary_builder_creates_audio_track(self):
        """Test that primary builder creates matching audio track."""
        match = MockMatch("C:/Videos/test.mp4", 0.0, 5.0)
        matches = [MockMatchResult(primary=match)]
        config = MockConfig(output=MockOutputConfig())

        builder = PrimaryTrackBuilder(matches, config, 30.0)
        video_track, audio_track = builder.build(0)

        assert len(audio_track) == 1
        audio_clip = audio_track[0]
        assert isinstance(audio_clip, otio.schema.Clip)
        assert audio_clip.metadata.get("from_track") == "V1"

    def test_primary_builder_with_segment_offset(self):
        """Test primary builder with segment file offset."""
        # Simulate segment file name pattern: video_0120.mp4 (offset = 120.0s)
        match = MockMatch("C:/Videos/video_0120.mp4", 5.0, 10.0)
        match.voiceover_segment = MockVoiceoverSegment(0.0, 5.0, "Test voiceover text")
        match.video_segment = MockVideoSegment("C:/Videos/video_0120.mp4", 5.0, 10.0, "Test video text")

        matches = [MockMatchResult(primary=match)]
        config = MockConfig(output=MockOutputConfig())

        builder = PrimaryTrackBuilder(matches, config, 30.0)
        video_track, audio_track = builder.build(0)

        # Should create clip successfully
        assert len(video_track) == 1
        assert isinstance(video_track[0], otio.schema.Clip)

    def test_primary_builder_with_audio_first_resolution(self):
        """Test primary builder with audio-first mode resolution."""
        match = MockMatch("audio.mp3", 0.0, 5.0)
        match.voiceover_segment = MockVoiceoverSegment(0.0, 5.0, "Test voiceover text")
        match.video_segment = MockVideoSegment("audio.mp3", 0.0, 5.0, "Test video text")

        matches = [MockMatchResult(primary=match)]
        config = MockConfig(output=MockOutputConfig())

        # Mock resolve function
        def mock_resolve(source_file, start_time):
            return "video.mp4", start_time + 1.0

        builder = PrimaryTrackBuilder(
            matches, config, 30.0,
            resolve_video_segment=mock_resolve
        )
        video_track, audio_track = builder.build(0)

        # Should create clip with resolved video
        assert len(video_track) == 1
        assert isinstance(video_track[0], otio.schema.Clip)


class TestEntityBuildersDelegation:
    """Test entity builders delegate to entity module functions."""

    def test_entity_image_builder_with_entity_images(self):
        """Test EntityImageTrackBuilder with entity images provided."""
        primary = MockMatch("primary.mp4", 0.0, 5.0)
        matches = [MockMatchResult(primary=primary)]
        config = MockConfig(output=MockOutputConfig())

        # Provide entity_images in kwargs
        entity_images = {"test_entity": [{"file": "test.jpg"}]}

        builder = EntityImageTrackBuilder(
            matches, config, 30.0,
            entity_images=entity_images
        )
        video_track, audio_track = builder.build(8)

        # Should create tracks (delegation to entity module)
        assert isinstance(video_track, otio.schema.Track)
        assert isinstance(audio_track, otio.schema.Track)

    def test_entity_video_builder_with_entity_videos(self):
        """Test EntityVideoTrackBuilder with entity videos provided."""
        primary = MockMatch("primary.mp4", 0.0, 5.0)
        matches = [MockMatchResult(primary=primary)]
        config = MockConfig(output=MockOutputConfig())

        # Provide entity_videos in kwargs
        entity_videos = {"test_entity": [{"file": "test.mp4"}]}

        builder = EntityVideoTrackBuilder(
            matches, config, 30.0,
            entity_videos=entity_videos
        )
        video_track, audio_track = builder.build(9)

        # Should create tracks (delegation to entity module)
        assert isinstance(video_track, otio.schema.Track)
        assert isinstance(audio_track, otio.schema.Track)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
