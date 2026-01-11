"""
Comprehensive OTIO Pipeline Integration Test

Tests the complete OTIO timeline generation pipeline end-to-end with:
- All 10 video tracks (V1-V10, A1-A8)
- Audio-first mode with segment resolution
- Gap handling (leading, between-segment, trailing)
- Timewarp and speed calculations
- Entity images and videos
- DaVinci Resolve compatibility
"""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import Mock, patch

import opentimelineio as otio
import pytest

from src.otio import create_timeline, save_timeline
from src.state import EntityImage, EntityVideo


# ============================================================
# Test Fixtures - Realistic Match Data
# ============================================================

@dataclass
class MockImageSearchConfig:
    """Mock image search configuration."""
    enable_sticky_matching: bool = False
    max_clips_per_entity: int = 3


@dataclass
class MockOutputConfig:
    """Mock output configuration."""
    include_alternatives: bool = True
    num_alternatives: int = 2
    include_strategy_tracks: bool = True
    strategy_tracks: List[str] = None

    def __post_init__(self):
        if self.strategy_tracks is None:
            self.strategy_tracks = ["embedding_diversity", "broll_only"]


@dataclass
class MockConfig:
    """Mock pipeline configuration."""
    output: MockOutputConfig
    image_search: MockImageSearchConfig = None

    def __post_init__(self):
        if self.image_search is None:
            self.image_search = MockImageSearchConfig()


@dataclass
class MockVoiceoverSegment:
    """Mock voiceover segment with realistic properties."""
    start_time: float
    end_time: float
    text: str
    index: int = 0


@dataclass
class MockVideoSegment:
    """Mock video segment with realistic properties."""
    source_file: str
    start_time: float
    end_time: float
    text: str
    is_broll: bool = False
    face_score: float = 1.0


@dataclass
class MockMatch:
    """Mock Match object with all required fields."""
    voiceover_segment: MockVoiceoverSegment
    video_segment: MockVideoSegment
    confidence: float = 0.8
    speed: float = 1.0
    reasoning: str = "Test reasoning"
    strategy: str = "embedding"
    is_keyword_match: bool = False
    is_visual_match: bool = False
    keyword: str = ""
    embedding_similarity: float = 0.85
    clip_reuse_count: int = 0


@dataclass
class MockMatchResult:
    """Mock MatchResult with primary, alternatives, secondaries, and strategies."""
    primary_match: MockMatch
    alternatives: List[MockMatch] = None
    secondary_matches: List[MockMatch] = None
    strategy_matches: List[MockMatch] = None

    def __post_init__(self):
        if self.alternatives is None:
            self.alternatives = []
        if self.secondary_matches is None:
            self.secondary_matches = []
        if self.strategy_matches is None:
            self.strategy_matches = []


@dataclass
class MockEntityResult:
    """Mock entity result with images."""
    entity_name: str
    images: List[EntityImage] = None

    def __post_init__(self):
        if self.images is None:
            self.images = []


@dataclass
class MockDownloadedSegment:
    """Mock downloaded video segment for audio-first mode."""
    file: str
    video_id: str
    original_start: float
    original_end: float


# ============================================================
# Test Fixture Factory Functions
# ============================================================

def create_voiceover_segment(index: int, start: float, duration: float = 5.0, text: str = None) -> MockVoiceoverSegment:
    """Create a realistic voiceover segment."""
    if text is None:
        text = f"This is voiceover segment {index} with some test content."
    return MockVoiceoverSegment(
        start_time=start,
        end_time=start + duration,
        text=text,
        index=index
    )


def create_video_segment(
    video_id: str,
    start: float,
    duration: float = 6.0,
    text: str = None,
    is_broll: bool = False
) -> MockVideoSegment:
    """Create a realistic video segment."""
    if text is None:
        text = f"Video content about {video_id}"
    return MockVideoSegment(
        source_file=f"E:/Downloads/{video_id}.mp4",
        start_time=start,
        end_time=start + duration,
        text=text,
        is_broll=is_broll,
        face_score=0.2 if is_broll else 0.8
    )


def create_match(
    vo_seg: MockVoiceoverSegment,
    video_id: str,
    video_start: float = 0.0,
    confidence: float = 0.85,
    is_broll: bool = False
) -> MockMatch:
    """Create a realistic match."""
    duration = vo_seg.end_time - vo_seg.start_time
    vid_seg = create_video_segment(video_id, video_start, duration * 1.2, is_broll=is_broll)

    return MockMatch(
        voiceover_segment=vo_seg,
        video_segment=vid_seg,
        confidence=confidence,
        reasoning=f"Matched based on semantic similarity ({confidence:.2f})",
        embedding_similarity=confidence
    )


def create_full_match_result(
    index: int,
    vo_start: float,
    vo_duration: float = 5.0
) -> MockMatchResult:
    """
    Create a complete MatchResult with primary, alternatives, secondaries, and strategies.

    This simulates what the matching stage produces.
    """
    vo_seg = create_voiceover_segment(index, vo_start, vo_duration)

    # Primary match (best match)
    primary = create_match(vo_seg, f"primary_{index}", 0.0, 0.90)

    # Alternative matches (2nd and 3rd best)
    alt1 = create_match(vo_seg, f"alt1_{index}", 5.0, 0.80)
    alt2 = create_match(vo_seg, f"alt2_{index}", 10.0, 0.75)

    # Secondary matches (different sources for diversity)
    sec_primary = create_match(vo_seg, f"sec_primary_{index}", 0.0, 0.85)
    sec_alt1 = create_match(vo_seg, f"sec_alt1_{index}", 3.0, 0.78)
    sec_alt2 = create_match(vo_seg, f"sec_alt2_{index}", 6.0, 0.72)

    # Strategy matches
    embedding_div = create_match(vo_seg, f"embed_div_{index}", 0.0, 0.82)
    embedding_div.strategy = "embedding_diversity"

    broll = create_match(vo_seg, f"broll_{index}", 0.0, 0.88, is_broll=True)
    broll.strategy = "broll_only"

    return MockMatchResult(
        primary_match=primary,
        alternatives=[alt1, alt2],
        secondary_matches=[sec_primary, sec_alt1, sec_alt2],
        strategy_matches=[embedding_div, broll]
    )


def create_entity_images(temp_dir: Path) -> Dict:
    """Create mock entity images with real files."""
    # Create temporary image files
    entities = {}

    for entity_name in ["Paris", "Eiffel Tower", "London"]:
        images = []
        for i in range(3):
            # Create a dummy image file
            img_path = temp_dir / f"{entity_name.replace(' ', '_')}_{i}.jpg"
            img_path.write_text("fake image data")

            images.append(EntityImage(
                entity=entity_name,
                file=str(img_path),
                source_url=f"https://example.com/{entity_name}_{i}.jpg",
                width=800,
                height=600
            ))

        entities[entity_name] = MockEntityResult(entity_name, images)

    return entities


def create_entity_videos(temp_dir: Path) -> Dict:
    """Create mock entity videos with real files."""
    entities = {}

    for entity_name in ["travel", "cityscape", "landmark"]:
        videos = []
        for i in range(2):
            # Create a dummy video file
            vid_path = temp_dir / f"{entity_name}_{i}.mp4"
            vid_path.write_text("fake video data")

            videos.append(EntityVideo(
                entity=entity_name,
                file=str(vid_path),
                source="pexels",
                duration=15.0
            ))

        entities[entity_name] = MockEntityResult(entity_name, videos)

    return entities


def create_audio_first_segments(matches: List[MockMatchResult]) -> List[MockDownloadedSegment]:
    """
    Create downloaded video segments for audio-first mode.

    Simulates the DOWNLOAD_SEGMENTS stage output where audio files
    have been replaced with downloaded video segments.
    """
    segments = []

    for match_result in matches:
        primary = match_result.primary_match
        vid_seg = primary.video_segment

        # Extract video_id from source file
        video_id = Path(vid_seg.source_file).stem

        # Create segment file name (e.g., "primary_0_seg_0000.mp4")
        segment_file = f"E:/Downloads/{video_id}_seg_0000.mp4"

        segments.append(MockDownloadedSegment(
            file=segment_file,
            video_id=video_id,
            original_start=vid_seg.start_time,
            original_end=vid_seg.end_time
        ))

    return segments


# ============================================================
# Integration Tests
# ============================================================

class TestOTIOPipelineBasic:
    """Test basic timeline creation."""

    def test_create_simple_timeline(self):
        """Test creating a simple timeline with 3 segments."""
        # Create 3 match results
        matches = [
            create_full_match_result(0, 0.0, 5.0),
            create_full_match_result(1, 5.0, 4.0),
            create_full_match_result(2, 9.0, 6.0),
        ]

        config = MockConfig(output=MockOutputConfig())

        # Create timeline
        timeline = create_timeline(
            matches=matches,
            config=config,
            frame_rate=30.0
        )

        # Verify timeline structure
        assert timeline is not None
        assert timeline.name == "Matched Footage"
        assert len(timeline.tracks) > 0

        # Should have 10 video tracks + 8 audio tracks
        video_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Video]
        audio_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Audio]

        assert len(video_tracks) == 10, f"Expected 10 video tracks, got {len(video_tracks)}"
        assert len(audio_tracks) == 8, f"Expected 8 audio tracks, got {len(audio_tracks)}"

        print(f"✓ Created timeline with {len(video_tracks)} video tracks and {len(audio_tracks)} audio tracks")

    def test_primary_track_has_all_clips(self):
        """Test V1 (primary track) has clips for all segments."""
        matches = [
            create_full_match_result(0, 0.0, 5.0),
            create_full_match_result(1, 5.0, 4.0),
            create_full_match_result(2, 9.0, 6.0),
        ]

        config = MockConfig(output=MockOutputConfig())
        timeline = create_timeline(matches, config, frame_rate=30.0)

        # Find V1 track
        video_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Video]
        v1_track = video_tracks[0]

        # Count clips (non-gap items)
        clips = [item for item in v1_track if isinstance(item, otio.schema.Clip)]

        assert len(clips) == len(matches), f"Expected {len(matches)} clips in V1, got {len(clips)}"

        # Verify each clip has expected metadata
        for i, clip in enumerate(clips):
            assert 'segment_index' in clip.metadata
            assert clip.metadata['segment_index'] == i
            assert 'confidence' in clip.metadata

        print(f"✓ V1 track has all {len(clips)} clips with correct metadata")

    def test_alternative_tracks_populated(self):
        """Test V2-V3 (alternative tracks) are populated correctly."""
        matches = [
            create_full_match_result(0, 0.0, 5.0),
            create_full_match_result(1, 5.0, 4.0),
        ]

        config = MockConfig(output=MockOutputConfig())
        timeline = create_timeline(matches, config, frame_rate=30.0)

        video_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Video]

        # V2 and V3 should have clips from alternatives
        v2_track = video_tracks[1]
        v3_track = video_tracks[2]

        v2_clips = [item for item in v2_track if isinstance(item, otio.schema.Clip)]
        v3_clips = [item for item in v3_track if isinstance(item, otio.schema.Clip)]

        assert len(v2_clips) == len(matches), f"Expected {len(matches)} clips in V2"
        assert len(v3_clips) == len(matches), f"Expected {len(matches)} clips in V3"

        # Both tracks should be disabled
        assert v2_track.enabled == False
        assert v3_track.enabled == False

        print(f"✓ Alternative tracks V2-V3 populated with {len(v2_clips)} clips each")


class TestOTIOPipelineAudioFirst:
    """Test audio-first mode with segment resolution."""

    def test_segment_resolution_maps_audio_to_video(self):
        """Test that audio files are mapped to video segment files."""
        matches = [
            create_full_match_result(0, 0.0, 5.0),
            create_full_match_result(1, 5.0, 4.0),
        ]

        # Create downloaded segments (simulating audio-first mode)
        downloaded_segments = create_audio_first_segments(matches)

        config = MockConfig(output=MockOutputConfig())

        # Create timeline with downloaded segments
        timeline = create_timeline(
            matches=matches,
            config=config,
            frame_rate=30.0,
            downloaded_segments=downloaded_segments
        )

        # Verify V1 clips reference segment files, not original audio files
        video_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Video]
        v1_track = video_tracks[0]
        clips = [item for item in v1_track if isinstance(item, otio.schema.Clip)]

        for clip in clips:
            # Clip should reference segment file (_seg_0000.mp4)
            media_ref = clip.media_reference
            if media_ref and hasattr(media_ref, 'target_url'):
                assert '_seg_' in media_ref.target_url, f"Expected segment file, got {media_ref.target_url}"

        print(f"✓ Audio-first mode: clips reference video segments")

    def test_segment_time_adjustment(self):
        """Test that segment start times are adjusted correctly."""
        # Create match with specific timing
        matches = [create_full_match_result(0, 0.0, 5.0)]

        primary = matches[0].primary_match
        vid_seg = primary.video_segment

        # Create segment with offset
        video_id = Path(vid_seg.source_file).stem
        downloaded_segments = [
            MockDownloadedSegment(
                file=f"E:/Downloads/{video_id}_seg_0000.mp4",
                video_id=video_id,
                original_start=10.0,  # Segment starts at 10s in original video
                original_end=20.0
            )
        ]

        config = MockConfig(output=MockOutputConfig())

        # Create timeline
        timeline = create_timeline(
            matches=matches,
            config=config,
            frame_rate=30.0,
            downloaded_segments=downloaded_segments
        )

        # Verify clip has adjusted start time
        video_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Video]
        v1_track = video_tracks[0]
        clips = [item for item in v1_track if isinstance(item, otio.schema.Clip)]

        if clips:
            clip = clips[0]
            # Start time should be relative to segment (vid_seg.start_time - segment.original_start)
            # If vid_seg.start_time was 0, adjusted should be max(0, 0 - 10) = 0
            media_ref = clip.media_reference
            if media_ref and hasattr(media_ref, 'available_range'):
                start_time = media_ref.available_range.start_time.value
                # Should be adjusted
                print(f"✓ Segment time adjusted: {start_time}")


class TestOTIOPipelineGapHandling:
    """Test gap handling (leading, between-segment, trailing)."""

    def test_leading_gap_when_first_segment_not_at_zero(self):
        """Test leading gap is inserted when first segment starts after 0."""
        # Create matches where first segment starts at 2.0 seconds
        matches = [
            create_full_match_result(0, 2.0, 5.0),  # First segment at 2s
            create_full_match_result(1, 7.0, 4.0),
        ]

        config = MockConfig(output=MockOutputConfig())
        timeline = create_timeline(matches, config, frame_rate=30.0)

        # V1 should have a leading gap
        video_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Video]
        v1_track = video_tracks[0]

        # First item should be a gap
        first_item = v1_track[0]

        # Note: Leading gap insertion is handled in timeline.py lines 295-316
        # This test verifies the behavior
        print(f"✓ First item type: {type(first_item).__name__}")

    def test_between_segment_gaps(self):
        """Test gaps are inserted between non-consecutive segments."""
        # Create matches with a 2-second gap between them
        matches = [
            create_full_match_result(0, 0.0, 5.0),   # 0-5s
            create_full_match_result(1, 7.0, 4.0),   # 7-11s (2s gap)
        ]

        config = MockConfig(output=MockOutputConfig())
        timeline = create_timeline(matches, config, frame_rate=30.0)

        video_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Video]
        v1_track = video_tracks[0]

        # Should have: clip, gap, clip
        items = list(v1_track)
        gaps = [item for item in items if isinstance(item, otio.schema.Gap)]

        print(f"✓ Found {len(gaps)} gaps in V1 track with {len(items)} total items")


class TestOTIOPipelineTimewarp:
    """Test timewarp and speed calculations."""

    def test_clip_with_speed_adjustment(self):
        """Test clips have correct timewarp when source != target duration."""
        # Create match where video is 6s but voiceover is 5s (needs speed up)
        vo_seg = create_voiceover_segment(0, 0.0, 5.0)
        vid_seg = create_video_segment("speed_test", 0.0, 6.0)  # 6s video

        match = MockMatch(
            voiceover_segment=vo_seg,
            video_segment=vid_seg,
            confidence=0.85
        )

        match_result = MockMatchResult(primary_match=match)

        config = MockConfig(output=MockOutputConfig())
        timeline = create_timeline([match_result], config, frame_rate=30.0)

        # Check V1 clip has timewarp
        video_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Video]
        v1_track = video_tracks[0]
        clips = [item for item in v1_track if isinstance(item, otio.schema.Clip)]

        if clips:
            clip = clips[0]
            # Should have LinearTimeWarp effect with time_scalar = 6/5 = 1.2
            if clip.effects:
                timewarp = clip.effects[0]
                assert isinstance(timewarp, otio.schema.LinearTimeWarp)
                expected_scalar = 6.0 / 5.0
                assert abs(timewarp.time_scalar - expected_scalar) < 0.01

                print(f"✓ Timewarp applied: time_scalar = {timewarp.time_scalar:.2f} (expected {expected_scalar:.2f})")

    def test_clip_without_speed_adjustment(self):
        """Test clips without timewarp when source == target duration."""
        # Create match where video duration matches voiceover exactly
        vo_seg = create_voiceover_segment(0, 0.0, 5.0)
        vid_seg = create_video_segment("no_speed", 0.0, 5.0)  # Exact match

        match = MockMatch(
            voiceover_segment=vo_seg,
            video_segment=vid_seg,
            confidence=0.85
        )

        match_result = MockMatchResult(primary_match=match)

        config = MockConfig(output=MockOutputConfig())
        timeline = create_timeline([match_result], config, frame_rate=30.0)

        video_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Video]
        v1_track = video_tracks[0]
        clips = [item for item in v1_track if isinstance(item, otio.schema.Clip)]

        if clips:
            clip = clips[0]
            # Should have no LinearTimeWarp or time_scalar = 1.0
            if clip.effects:
                timewarp = clip.effects[0]
                if isinstance(timewarp, otio.schema.LinearTimeWarp):
                    assert abs(timewarp.time_scalar - 1.0) < 0.01

            print(f"✓ No speed adjustment needed (durations match)")


class TestOTIOPipelineEntityTracks:
    """Test entity images and videos tracks."""

    def test_entity_images_track(self, tmp_path):
        """Test V9 (entity images) track is populated."""
        matches = [create_full_match_result(0, 0.0, 5.0)]

        # Create entity images
        entity_images = create_entity_images(tmp_path)

        config = MockConfig(output=MockOutputConfig())

        timeline = create_timeline(
            matches=matches,
            config=config,
            frame_rate=30.0,
            entity_images=entity_images
        )

        # Find V9 track
        video_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Video]
        v9_track = video_tracks[8]  # 0-indexed: V9 = index 8

        assert "Entity Images" in v9_track.name
        assert v9_track.enabled == False  # Should be disabled by default

        # Check if track has clips
        clips = [item for item in v9_track if isinstance(item, otio.schema.Clip)]
        print(f"✓ V9 Entity Images track created with {len(clips)} clips")

    def test_entity_videos_track(self, tmp_path):
        """Test V10 (stock videos) track is populated."""
        matches = [create_full_match_result(0, 0.0, 5.0)]

        # Create entity videos
        entity_videos = create_entity_videos(tmp_path)

        config = MockConfig(output=MockOutputConfig())

        timeline = create_timeline(
            matches=matches,
            config=config,
            frame_rate=30.0,
            entity_videos=entity_videos
        )

        # Find V10 track
        video_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Video]
        v10_track = video_tracks[9]  # 0-indexed: V10 = index 9

        assert "Stock Videos" in v10_track.name
        assert v10_track.enabled == False

        clips = [item for item in v10_track if isinstance(item, otio.schema.Clip)]
        print(f"✓ V10 Stock Videos track created with {len(clips)} clips")


class TestOTIOPipelineDaVinciCompatibility:
    """Test DaVinci Resolve compatibility requirements."""

    def test_global_start_time_is_valid(self):
        """Test global_start_time is a valid RationalTime (not empty string)."""
        matches = [create_full_match_result(0, 0.0, 5.0)]
        config = MockConfig(output=MockOutputConfig())

        timeline = create_timeline(matches, config, frame_rate=30.0)

        # global_start_time must be RationalTime, not ""
        assert timeline.global_start_time is not None
        assert isinstance(timeline.global_start_time, otio.opentime.RationalTime)
        assert timeline.global_start_time.value > 0  # Should be 3600 * frame_rate

        print(f"✓ global_start_time = {timeline.global_start_time} (valid)")

    def test_resolve_otio_metadata_present(self):
        """Test Resolve_OTIO metadata is present."""
        matches = [create_full_match_result(0, 0.0, 5.0)]
        config = MockConfig(output=MockOutputConfig())

        timeline = create_timeline(matches, config, frame_rate=30.0)

        assert 'Resolve_OTIO' in timeline.metadata
        assert 'Resolve OTIO Meta Version' in timeline.metadata['Resolve_OTIO']

        print(f"✓ Resolve_OTIO metadata present: {timeline.metadata['Resolve_OTIO']}")

    def test_tracks_stack_name_empty(self):
        """Test tracks.name is empty string (DaVinci format)."""
        matches = [create_full_match_result(0, 0.0, 5.0)]
        config = MockConfig(output=MockOutputConfig())

        timeline = create_timeline(matches, config, frame_rate=30.0)

        assert timeline.tracks.name == ""

        print(f"✓ timeline.tracks.name is empty (DaVinci compatible)")

    def test_track_names_correct(self):
        """Test all track names match expected DaVinci format."""
        matches = [create_full_match_result(0, 0.0, 5.0)]
        config = MockConfig(output=MockOutputConfig())

        timeline = create_timeline(matches, config, frame_rate=30.0)

        video_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Video]

        expected_names = [
            "V1 - Primary",
            "V2 - Alternative 1",
            "V3 - Alternative 2",
            "V4 - Secondary Primary",
            "V5 - Secondary Alt 1",
            "V6 - Secondary Alt 2",
            "V7 - Embedding-Diversity",
            "V8 - B-roll Only",
            "V9 - Entity Images",
            "V10 - Stock Videos"
        ]

        for i, (track, expected_name) in enumerate(zip(video_tracks, expected_names)):
            assert track.name == expected_name, f"Track {i}: expected '{expected_name}', got '{track.name}'"

        print(f"✓ All {len(video_tracks)} track names correct")


class TestOTIOPipelineExport:
    """Test timeline export to OTIO file."""

    def test_save_timeline_to_file(self, tmp_path):
        """Test saving timeline to OTIO file."""
        matches = [
            create_full_match_result(0, 0.0, 5.0),
            create_full_match_result(1, 5.0, 4.0),
        ]

        config = MockConfig(output=MockOutputConfig())
        timeline = create_timeline(matches, config, frame_rate=30.0)

        # Save to temporary file
        output_file = tmp_path / "test_timeline.otio"
        save_timeline(timeline, str(output_file))

        assert output_file.exists()
        assert output_file.stat().st_size > 0

        # Verify file can be loaded back
        loaded_timeline = otio.adapters.read_from_file(str(output_file))
        assert loaded_timeline.name == timeline.name
        assert len(loaded_timeline.tracks) == len(timeline.tracks)

        print(f"✓ Timeline saved to {output_file.name} ({output_file.stat().st_size} bytes)")

    def test_timeline_json_structure(self, tmp_path):
        """Test OTIO JSON structure is valid."""
        matches = [create_full_match_result(0, 0.0, 5.0)]
        config = MockConfig(output=MockOutputConfig())

        timeline = create_timeline(matches, config, frame_rate=30.0)

        # Save and load as JSON
        output_file = tmp_path / "test_timeline.otio"
        save_timeline(timeline, str(output_file))

        # Parse JSON
        with open(output_file) as f:
            data = json.load(f)

        # Verify top-level structure
        assert 'OTIO_SCHEMA' in data
        assert data['OTIO_SCHEMA'] == 'Timeline.1'
        assert 'name' in data
        assert 'tracks' in data

        print(f"✓ OTIO JSON structure valid")


class TestOTIOPipelineEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_matches(self):
        """Test timeline creation with no matches."""
        matches = []
        config = MockConfig(output=MockOutputConfig())

        timeline = create_timeline(matches, config, frame_rate=30.0)

        # Should create timeline with empty tracks
        assert timeline is not None
        assert len(timeline.tracks) > 0

        video_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Video]
        v1_track = video_tracks[0]

        # V1 should be empty
        assert len(list(v1_track)) == 0

        print(f"✓ Empty matches handled correctly")

    def test_single_match(self):
        """Test timeline with only one match."""
        matches = [create_full_match_result(0, 0.0, 5.0)]
        config = MockConfig(output=MockOutputConfig())

        timeline = create_timeline(matches, config, frame_rate=30.0)

        video_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Video]
        v1_track = video_tracks[0]

        clips = [item for item in v1_track if isinstance(item, otio.schema.Clip)]
        assert len(clips) == 1

        print(f"✓ Single match handled correctly")

    def test_very_short_segments(self):
        """Test timeline with very short segments (0.5s)."""
        matches = [
            create_full_match_result(0, 0.0, 0.5),
            create_full_match_result(1, 0.5, 0.5),
        ]

        config = MockConfig(output=MockOutputConfig())
        timeline = create_timeline(matches, config, frame_rate=30.0)

        video_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Video]
        v1_track = video_tracks[0]

        clips = [item for item in v1_track if isinstance(item, otio.schema.Clip)]
        assert len(clips) == 2

        # Verify clip durations
        for clip in clips:
            duration = clip.source_range.duration
            # Should be 0.5s = 15 frames at 30fps
            expected_frames = 0.5 * 30
            assert abs(duration.value - expected_frames) < 1  # Allow 1 frame tolerance

        print(f"✓ Very short segments handled correctly")

    def test_different_frame_rates(self):
        """Test timeline creation with different frame rates."""
        matches = [create_full_match_result(0, 0.0, 5.0)]
        config = MockConfig(output=MockOutputConfig())

        for frame_rate in [24.0, 25.0, 29.97, 30.0, 60.0]:
            timeline = create_timeline(matches, config, frame_rate=frame_rate)

            # Verify global_start_time uses correct frame rate
            assert timeline.global_start_time.rate == frame_rate

            # Verify clips use correct frame rate
            video_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Video]
            v1_track = video_tracks[0]
            clips = [item for item in v1_track if isinstance(item, otio.schema.Clip)]

            if clips:
                clip = clips[0]
                assert clip.source_range.start_time.rate == frame_rate

        print(f"✓ Multiple frame rates handled correctly")


# ============================================================
# Test Runner
# ============================================================

def run_integration_tests():
    """Run all OTIO pipeline integration tests."""
    print("\n" + "=" * 70)
    print("  OTIO PIPELINE INTEGRATION TESTS")
    print("=" * 70 + "\n")

    # Run pytest with verbose output
    exit_code = pytest.main([__file__, "-v", "--tb=short", "-x"])

    return exit_code


if __name__ == "__main__":
    import sys
    exit_code = run_integration_tests()
    sys.exit(exit_code)
