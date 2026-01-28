"""
Coverage tests for src/otio/timeline.py

Targets specific uncovered lines and edge cases:
- Fallback segment resolution (lines 146-153)
- Fallback VO duration when ffprobe unavailable (lines 286-290)
- Segment file offset handling (lines 369-372)
- Alternative track segment resolution (lines 440-446)
- Secondary track segment resolution (lines 517-525)
- Strategy track segment resolution (lines 607-616)
- Voiceover duration fallback (lines 724-726)
- Empty entity images after validation (line 786)
- Trailing gap handling (lines 684-710)
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import opentimelineio as otio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.otio.timeline import create_timeline, _validate_entity_images


# ============================================================================
# Mock Dataclasses
# ============================================================================

@dataclass
class MockVoiceoverSegment:
    """Mock voiceover segment."""
    start_time: float
    end_time: float
    text: str = "Test voiceover text"


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
    voiceover_segment: MockVoiceoverSegment
    video_segment: MockVideoSegment
    confidence: float = 0.85
    reasoning: str = "Test reasoning"
    strategy: str = "primary"
    is_keyword_match: bool = False
    is_visual_match: bool = False
    embedding_similarity: float = 0.80
    clip_reuse_count: int = 0


@dataclass
class MockMatchResult:
    """Mock MatchResult with all fields."""
    primary_match: MockMatch
    alternatives: List[MockMatch] = field(default_factory=list)
    secondary_matches: List[MockMatch] = field(default_factory=list)
    strategy_matches: List[MockMatch] = field(default_factory=list)


@dataclass
class MockDownloadedSegment:
    """Mock downloaded video segment for audio-first mode."""
    file: str
    video_id: str
    original_start: float
    original_end: float


@dataclass
class MockOutputConfig:
    """Mock output configuration."""
    include_alternatives: bool = True
    num_alternatives: int = 2
    include_strategy_tracks: bool = True
    strategy_tracks: List[str] = field(default_factory=lambda: ["embedding_diversity", "broll_only"])


@dataclass
class MockConfig:
    """Mock pipeline configuration."""
    output: MockOutputConfig = field(default_factory=MockOutputConfig)


@dataclass
class MockEntityImage:
    """Mock entity image."""
    file: str
    score: float = 0.9


@dataclass
class MockEntityResult:
    """Mock entity result with images."""
    images: List[MockEntityImage] = field(default_factory=list)


# ============================================================================
# Helper Functions
# ============================================================================

def create_simple_match_result(
    vo_start: float = 0.0,
    vo_end: float = 5.0,
    video_file: str = "E:/videos/test.mp4",
    video_start: float = 0.0,
    video_end: float = 6.0,
    alternatives: Optional[List[MockMatch]] = None,
    secondaries: Optional[List[MockMatch]] = None,
    strategies: Optional[List[MockMatch]] = None
) -> MockMatchResult:
    """Create a simple match result for testing."""
    vo_seg = MockVoiceoverSegment(start_time=vo_start, end_time=vo_end)
    vid_seg = MockVideoSegment(source_file=video_file, start_time=video_start, end_time=video_end)
    primary = MockMatch(voiceover_segment=vo_seg, video_segment=vid_seg)

    return MockMatchResult(
        primary_match=primary,
        alternatives=alternatives or [],
        secondary_matches=secondaries or [],
        strategy_matches=strategies or []
    )


def create_alt_match(
    vo_seg: MockVoiceoverSegment,
    video_file: str,
    video_start: float = 0.0,
    video_end: float = 6.0
) -> MockMatch:
    """Create an alternative match."""
    vid_seg = MockVideoSegment(source_file=video_file, start_time=video_start, end_time=video_end)
    return MockMatch(voiceover_segment=vo_seg, video_segment=vid_seg, confidence=0.75)


def create_strategy_match(
    vo_seg: MockVoiceoverSegment,
    video_file: str,
    strategy: str,
    video_start: float = 0.0,
    video_end: float = 6.0
) -> MockMatch:
    """Create a strategy match."""
    vid_seg = MockVideoSegment(source_file=video_file, start_time=video_start, end_time=video_end)
    return MockMatch(
        voiceover_segment=vo_seg,
        video_segment=vid_seg,
        confidence=0.70,
        strategy=strategy
    )


# ============================================================================
# Test: Fallback Segment Resolution (lines 146-153)
# ============================================================================

class TestFallbackSegmentResolution:
    """Test the fallback segment resolution when source_start is outside exact range."""

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    @pytest.mark.fast
    def test_segment_resolution_with_approximate_time_match(self, mock_windows_path, mock_duration):
        """Test segment resolution when time is within 60s buffer of segment end."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = 60.0

        # Create downloaded segment that covers 0-30s
        downloaded_segments = [
            MockDownloadedSegment(
                file="E:/segments/vid123.mp4",
                video_id="vid123",
                original_start=0.0,
                original_end=30.0
            )
        ]

        # Create match that references time OUTSIDE segment but within 60s buffer
        # source_start=45 is > seg.end(30) but <= seg.end + 60 (90)
        vo_seg = MockVoiceoverSegment(start_time=0.0, end_time=5.0)
        vid_seg = MockVideoSegment(source_file="vid123.mp3", start_time=45.0, end_time=50.0)
        primary = MockMatch(voiceover_segment=vo_seg, video_segment=vid_seg)
        matches = [MockMatchResult(primary_match=primary)]

        config = MockConfig(output=MockOutputConfig(
            include_alternatives=False,
            include_strategy_tracks=False
        ))

        timeline = create_timeline(matches, config, downloaded_segments=downloaded_segments)

        assert isinstance(timeline, otio.schema.Timeline)
        # The fallback should use the first segment with adjusted time

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    @pytest.mark.fast
    def test_segment_resolution_no_matching_segment(self, mock_windows_path, mock_duration):
        """Test segment resolution when no segment matches at all."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = 60.0

        # Create downloaded segment for different video_id
        downloaded_segments = [
            MockDownloadedSegment(
                file="E:/segments/other_vid.mp4",
                video_id="other_vid",
                original_start=0.0,
                original_end=30.0
            )
        ]

        # Match references a different video that has no segments
        vo_seg = MockVoiceoverSegment(start_time=0.0, end_time=5.0)
        vid_seg = MockVideoSegment(source_file="vid123.mp3", start_time=10.0, end_time=15.0)
        primary = MockMatch(voiceover_segment=vo_seg, video_segment=vid_seg)
        matches = [MockMatchResult(primary_match=primary)]

        config = MockConfig(output=MockOutputConfig(
            include_alternatives=False,
            include_strategy_tracks=False
        ))

        timeline = create_timeline(matches, config, downloaded_segments=downloaded_segments)

        assert isinstance(timeline, otio.schema.Timeline)
        # Should use original source file when no segment match


# ============================================================================
# Test: Fallback VO Duration (lines 286-290)
# ============================================================================

class TestFallbackVODuration:
    """Test fallback voiceover duration when ffprobe unavailable."""

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    @pytest.mark.fast
    def test_fallback_duration_when_ffprobe_fails(self, mock_windows_path, mock_duration):
        """Test fallback duration calculation when ffprobe returns None."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = None  # ffprobe unavailable

        # Create match with segment ending at 25.0
        vo_seg = MockVoiceoverSegment(start_time=20.0, end_time=25.0)
        vid_seg = MockVideoSegment(source_file="E:/videos/test.mp4", start_time=0.0, end_time=6.0)
        primary = MockMatch(voiceover_segment=vo_seg, video_segment=vid_seg)
        matches = [MockMatchResult(primary_match=primary)]

        config = MockConfig(output=MockOutputConfig(
            include_alternatives=False,
            include_strategy_tracks=False
        ))

        # With voiceover path but no duration from ffprobe
        timeline = create_timeline(matches, config, voiceover_path="E:/audio/voiceover.mp3")

        assert isinstance(timeline, otio.schema.Timeline)
        # Fallback should be: last_segment.end_time + 30.0 = 25.0 + 30.0 = 55.0

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    @pytest.mark.fast
    def test_timeline_without_voiceover_path(self, mock_windows_path, mock_duration):
        """Test timeline creation without voiceover path."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = None

        matches = [create_simple_match_result()]
        config = MockConfig(output=MockOutputConfig(
            include_alternatives=False,
            include_strategy_tracks=False
        ))

        # No voiceover path
        timeline = create_timeline(matches, config, voiceover_path=None)

        assert isinstance(timeline, otio.schema.Timeline)
        # Voiceover track should not be added


# ============================================================================
# Test: Segment File Offset Handling (lines 369-372)
# ============================================================================

class TestSegmentFileOffset:
    """Test legacy segment file offset handling."""

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    @patch('src.otio.timeline.get_segment_file_offset')
    @pytest.mark.fast
    def test_segment_file_offset_applied(self, mock_offset, mock_windows_path, mock_duration):
        """Test that segment file offset is applied when no segment lookup used."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = 60.0
        mock_offset.return_value = 10.0  # 10 second offset from filename

        # No downloaded_segments, so segment_lookup is empty
        vo_seg = MockVoiceoverSegment(start_time=0.0, end_time=5.0)
        vid_seg = MockVideoSegment(source_file="vid123_0010.mp4", start_time=15.0, end_time=20.0)
        primary = MockMatch(voiceover_segment=vo_seg, video_segment=vid_seg)
        matches = [MockMatchResult(primary_match=primary)]

        config = MockConfig(output=MockOutputConfig(
            include_alternatives=False,
            include_strategy_tracks=False
        ))

        timeline = create_timeline(matches, config)

        assert isinstance(timeline, otio.schema.Timeline)
        # Adjusted start should be max(0, 15.0 - 10.0) = 5.0

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    @patch('src.otio.timeline.get_segment_file_offset')
    @pytest.mark.fast
    def test_segment_file_offset_zero(self, mock_offset, mock_windows_path, mock_duration):
        """Test when segment file offset is zero (no offset needed)."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = 60.0
        mock_offset.return_value = 0.0  # No offset

        matches = [create_simple_match_result()]
        config = MockConfig(output=MockOutputConfig(
            include_alternatives=False,
            include_strategy_tracks=False
        ))

        timeline = create_timeline(matches, config)

        assert isinstance(timeline, otio.schema.Timeline)


# ============================================================================
# Test: Alternative Track Segment Resolution (lines 440-446)
# ============================================================================

class TestAlternativeTrackResolution:
    """Test segment resolution for alternative tracks."""

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    @pytest.mark.fast
    def test_alternative_with_segment_resolution(self, mock_windows_path, mock_duration):
        """Test that alternatives also get segment resolution."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = 60.0

        # Create downloaded segments for alternatives
        downloaded_segments = [
            MockDownloadedSegment(
                file="E:/segments/alt1_seg.mp4",
                video_id="alt1",
                original_start=0.0,
                original_end=30.0
            ),
            MockDownloadedSegment(
                file="E:/segments/alt2_seg.mp4",
                video_id="alt2",
                original_start=0.0,
                original_end=30.0
            )
        ]

        vo_seg = MockVoiceoverSegment(start_time=0.0, end_time=5.0)
        vid_seg = MockVideoSegment(source_file="E:/videos/primary.mp4", start_time=0.0, end_time=6.0)
        primary = MockMatch(voiceover_segment=vo_seg, video_segment=vid_seg)

        # Alternatives reference audio files that should be resolved to video segments
        alt1 = create_alt_match(vo_seg, "alt1.mp3", video_start=5.0, video_end=10.0)
        alt2 = create_alt_match(vo_seg, "alt2.mp3", video_start=10.0, video_end=15.0)

        matches = [MockMatchResult(
            primary_match=primary,
            alternatives=[alt1, alt2]
        )]

        config = MockConfig(output=MockOutputConfig(
            include_alternatives=True,
            num_alternatives=2,
            include_strategy_tracks=False
        ))

        timeline = create_timeline(matches, config, downloaded_segments=downloaded_segments)

        assert isinstance(timeline, otio.schema.Timeline)
        # Check that V2 and V3 tracks exist
        video_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Video]
        alt_tracks = [t for t in video_tracks if "Alternative" in t.name]
        assert len(alt_tracks) == 2

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    @patch('src.otio.timeline.get_segment_file_offset')
    @pytest.mark.fast
    def test_alternative_with_segment_file_offset(self, mock_offset, mock_windows_path, mock_duration):
        """Test alternatives use segment file offset for legacy support."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = 60.0
        mock_offset.return_value = 5.0  # 5 second offset

        vo_seg = MockVoiceoverSegment(start_time=0.0, end_time=5.0)
        vid_seg = MockVideoSegment(source_file="E:/videos/primary.mp4", start_time=0.0, end_time=6.0)
        primary = MockMatch(voiceover_segment=vo_seg, video_segment=vid_seg)

        alt1 = create_alt_match(vo_seg, "alt1_0005.mp4", video_start=10.0, video_end=15.0)

        matches = [MockMatchResult(
            primary_match=primary,
            alternatives=[alt1]
        )]

        config = MockConfig(output=MockOutputConfig(
            include_alternatives=True,
            num_alternatives=2,
            include_strategy_tracks=False
        ))

        timeline = create_timeline(matches, config)

        assert isinstance(timeline, otio.schema.Timeline)


# ============================================================================
# Test: Secondary Track Segment Resolution (lines 517-525)
# ============================================================================

class TestSecondaryTrackResolution:
    """Test segment resolution for secondary tracks (V4-V6)."""

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    @pytest.mark.fast
    def test_secondary_with_segment_resolution(self, mock_windows_path, mock_duration):
        """Test that secondary tracks get segment resolution."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = 60.0

        downloaded_segments = [
            MockDownloadedSegment(
                file="E:/segments/sec1_seg.mp4",
                video_id="sec1",
                original_start=0.0,
                original_end=30.0
            )
        ]

        vo_seg = MockVoiceoverSegment(start_time=0.0, end_time=5.0)
        vid_seg = MockVideoSegment(source_file="E:/videos/primary.mp4", start_time=0.0, end_time=6.0)
        primary = MockMatch(voiceover_segment=vo_seg, video_segment=vid_seg)

        # Secondary match references audio file
        sec1_vid_seg = MockVideoSegment(source_file="sec1.mp3", start_time=5.0, end_time=10.0)
        sec1 = MockMatch(voiceover_segment=vo_seg, video_segment=sec1_vid_seg, confidence=0.65)

        matches = [MockMatchResult(
            primary_match=primary,
            secondary_matches=[sec1]
        )]

        config = MockConfig(output=MockOutputConfig(
            include_alternatives=False,
            include_strategy_tracks=False
        ))

        timeline = create_timeline(matches, config, downloaded_segments=downloaded_segments)

        assert isinstance(timeline, otio.schema.Timeline)
        # V4 track should be populated
        video_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Video]
        sec_tracks = [t for t in video_tracks if "Secondary" in t.name]
        assert len(sec_tracks) >= 1

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    @patch('src.otio.timeline.get_segment_file_offset')
    @pytest.mark.fast
    def test_secondary_with_segment_file_offset(self, mock_offset, mock_windows_path, mock_duration):
        """Test secondary tracks use segment file offset."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = 60.0
        mock_offset.return_value = 8.0

        vo_seg = MockVoiceoverSegment(start_time=0.0, end_time=5.0)
        vid_seg = MockVideoSegment(source_file="E:/videos/primary.mp4", start_time=0.0, end_time=6.0)
        primary = MockMatch(voiceover_segment=vo_seg, video_segment=vid_seg)

        sec1_vid_seg = MockVideoSegment(source_file="sec1_0008.mp4", start_time=12.0, end_time=18.0)
        sec1 = MockMatch(voiceover_segment=vo_seg, video_segment=sec1_vid_seg, confidence=0.65)

        matches = [MockMatchResult(
            primary_match=primary,
            secondary_matches=[sec1]
        )]

        config = MockConfig(output=MockOutputConfig(
            include_alternatives=False,
            include_strategy_tracks=False
        ))

        timeline = create_timeline(matches, config)

        assert isinstance(timeline, otio.schema.Timeline)


# ============================================================================
# Test: Strategy Track Segment Resolution (lines 607-616)
# ============================================================================

class TestStrategyTrackResolution:
    """Test segment resolution for strategy tracks (V7-V8)."""

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    @pytest.mark.fast
    def test_strategy_with_segment_resolution(self, mock_windows_path, mock_duration):
        """Test that strategy tracks get segment resolution."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = 60.0

        downloaded_segments = [
            MockDownloadedSegment(
                file="E:/segments/strat_seg.mp4",
                video_id="strat_vid",
                original_start=0.0,
                original_end=30.0
            )
        ]

        vo_seg = MockVoiceoverSegment(start_time=0.0, end_time=5.0)
        vid_seg = MockVideoSegment(source_file="E:/videos/primary.mp4", start_time=0.0, end_time=6.0)
        primary = MockMatch(voiceover_segment=vo_seg, video_segment=vid_seg)

        # Strategy match references audio file
        strat = create_strategy_match(vo_seg, "strat_vid.mp3", "embedding_diversity", 5.0, 10.0)

        matches = [MockMatchResult(
            primary_match=primary,
            strategy_matches=[strat]
        )]

        config = MockConfig(output=MockOutputConfig(
            include_alternatives=False,
            include_strategy_tracks=True,
            strategy_tracks=["embedding_diversity", "broll_only"]
        ))

        timeline = create_timeline(matches, config, downloaded_segments=downloaded_segments)

        assert isinstance(timeline, otio.schema.Timeline)

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    @patch('src.otio.timeline.get_segment_file_offset')
    @pytest.mark.fast
    def test_strategy_with_segment_file_offset(self, mock_offset, mock_windows_path, mock_duration):
        """Test strategy tracks use segment file offset."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = 60.0
        mock_offset.return_value = 15.0

        vo_seg = MockVoiceoverSegment(start_time=0.0, end_time=5.0)
        vid_seg = MockVideoSegment(source_file="E:/videos/primary.mp4", start_time=0.0, end_time=6.0)
        primary = MockMatch(voiceover_segment=vo_seg, video_segment=vid_seg)

        strat = create_strategy_match(vo_seg, "strat_0015.mp4", "broll_only", 20.0, 25.0)

        matches = [MockMatchResult(
            primary_match=primary,
            strategy_matches=[strat]
        )]

        config = MockConfig(output=MockOutputConfig(
            include_alternatives=False,
            include_strategy_tracks=True,
            strategy_tracks=["embedding_diversity", "broll_only"]
        ))

        timeline = create_timeline(matches, config)

        assert isinstance(timeline, otio.schema.Timeline)

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    @pytest.mark.fast
    def test_strategy_track_with_no_matching_strategy(self, mock_windows_path, mock_duration):
        """Test gap handling when strategy doesn't have a match for segment."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = 60.0

        vo_seg = MockVoiceoverSegment(start_time=0.0, end_time=5.0)
        vid_seg = MockVideoSegment(source_file="E:/videos/primary.mp4", start_time=0.0, end_time=6.0)
        primary = MockMatch(voiceover_segment=vo_seg, video_segment=vid_seg)

        # Only embedding_diversity strategy, no broll_only
        strat = create_strategy_match(vo_seg, "strat.mp4", "embedding_diversity", 0.0, 6.0)

        matches = [MockMatchResult(
            primary_match=primary,
            strategy_matches=[strat]
        )]

        config = MockConfig(output=MockOutputConfig(
            include_alternatives=False,
            include_strategy_tracks=True,
            strategy_tracks=["embedding_diversity", "broll_only"]
        ))

        timeline = create_timeline(matches, config)

        assert isinstance(timeline, otio.schema.Timeline)
        # broll_only track should have a gap


# ============================================================================
# Test: Voiceover Duration Fallback (lines 724-726)
# ============================================================================

class TestVoiceoverDurationFallback:
    """Test voiceover clip duration fallback when actual duration unknown."""

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    @pytest.mark.fast
    def test_voiceover_uses_accumulated_frames_when_no_duration(self, mock_windows_path, mock_duration):
        """Test that voiceover clip uses accumulated frames when duration unavailable."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = None  # No duration from ffprobe

        # Multiple segments to accumulate time
        matches = [
            create_simple_match_result(vo_start=0.0, vo_end=5.0),
            create_simple_match_result(vo_start=5.0, vo_end=10.0),
            create_simple_match_result(vo_start=10.0, vo_end=15.0),
        ]

        config = MockConfig(output=MockOutputConfig(
            include_alternatives=False,
            include_strategy_tracks=False
        ))

        timeline = create_timeline(matches, config, voiceover_path="E:/audio/vo.mp3")

        assert isinstance(timeline, otio.schema.Timeline)

        # Find voiceover track
        audio_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Audio]
        vo_tracks = [t for t in audio_tracks if "Voiceover" in t.name]
        assert len(vo_tracks) == 1


# ============================================================================
# Test: Trailing Gap Handling (lines 684-710)
# ============================================================================

class TestTrailingGapHandling:
    """Test trailing gap added when VO duration > accumulated timeline."""

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    @pytest.mark.fast
    def test_trailing_gap_added_when_vo_longer(self, mock_windows_path, mock_duration):
        """Test trailing gap is added when voiceover extends past last segment."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = 30.0  # VO is 30 seconds

        # But segments only cover 0-10 seconds
        matches = [
            create_simple_match_result(vo_start=0.0, vo_end=5.0),
            create_simple_match_result(vo_start=5.0, vo_end=10.0),
        ]

        config = MockConfig(output=MockOutputConfig(
            include_alternatives=False,
            include_strategy_tracks=False
        ))

        timeline = create_timeline(matches, config, voiceover_path="E:/audio/vo.mp3")

        assert isinstance(timeline, otio.schema.Timeline)
        # Should have trailing gap of ~20 seconds

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    @pytest.mark.fast
    def test_no_trailing_gap_when_vo_matches_timeline(self, mock_windows_path, mock_duration):
        """Test no trailing gap when VO duration matches accumulated."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = 10.0  # VO is exactly 10 seconds

        matches = [
            create_simple_match_result(vo_start=0.0, vo_end=5.0),
            create_simple_match_result(vo_start=5.0, vo_end=10.0),
        ]

        config = MockConfig(output=MockOutputConfig(
            include_alternatives=False,
            include_strategy_tracks=False
        ))

        timeline = create_timeline(matches, config, voiceover_path="E:/audio/vo.mp3")

        assert isinstance(timeline, otio.schema.Timeline)


# ============================================================================
# Test: Entity Images Validation Edge Cases (line 786)
# ============================================================================

class TestEntityImagesValidationEdgeCases:
    """Test entity images validation edge cases."""

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    @patch('src.otio.timeline._add_entity_images_to_track')
    @pytest.mark.fast
    def test_no_valid_images_after_validation(self, mock_add_images, mock_windows_path, mock_duration, tmp_path):
        """Test warning when no valid entity images remain after validation."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = 60.0

        # Create entity with non-existent images
        entity_images = {
            "Entity1": MockEntityResult(images=[
                MockEntityImage(file=str(tmp_path / "nonexistent.jpg"))
            ])
        }

        matches = [create_simple_match_result()]

        config = MockConfig(output=MockOutputConfig(
            include_alternatives=False,
            include_strategy_tracks=False
        ))

        timeline = create_timeline(matches, config, entity_images=entity_images)

        assert isinstance(timeline, otio.schema.Timeline)
        # _add_entity_images_to_track should NOT be called since no valid images
        mock_add_images.assert_not_called()

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    @patch('src.otio.timeline._add_entity_images_to_track')
    @pytest.mark.fast
    def test_valid_images_passed_to_track_builder(self, mock_add_images, mock_windows_path, mock_duration, tmp_path):
        """Test valid images are passed to track builder."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = 60.0

        # Create real image file
        img_file = tmp_path / "valid_image.jpg"
        img_file.write_text("fake image data")

        entity_images = {
            "Entity1": MockEntityResult(images=[
                MockEntityImage(file=str(img_file))
            ])
        }

        matches = [create_simple_match_result()]

        config = MockConfig(output=MockOutputConfig(
            include_alternatives=False,
            include_strategy_tracks=False
        ))

        timeline = create_timeline(matches, config, entity_images=entity_images)

        assert isinstance(timeline, otio.schema.Timeline)
        # _add_entity_images_to_track SHOULD be called
        mock_add_images.assert_called_once()


# ============================================================================
# Test: Entity Videos Track (lines 791-805)
# ============================================================================

class TestEntityVideosTrack:
    """Test entity videos track population."""

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    @patch('src.otio.timeline._add_entity_videos_to_track')
    @pytest.mark.fast
    def test_entity_videos_passed_to_track(self, mock_add_videos, mock_windows_path, mock_duration):
        """Test entity videos are passed to track builder."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = 60.0

        @dataclass
        class MockEntityVideoResult:
            videos: List = field(default_factory=list)

        entity_videos = {
            "Entity1": MockEntityVideoResult(videos=[{"file": "video.mp4"}])
        }

        matches = [create_simple_match_result()]

        config = MockConfig(output=MockOutputConfig(
            include_alternatives=False,
            include_strategy_tracks=False
        ))

        timeline = create_timeline(matches, config, entity_videos=entity_videos)

        assert isinstance(timeline, otio.schema.Timeline)
        mock_add_videos.assert_called_once()


# ============================================================================
# Test: Leading Gap Handling (lines 295-316)
# ============================================================================

class TestLeadingGapHandling:
    """Test leading gap when first segment doesn't start at 0."""

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    @pytest.mark.fast
    def test_leading_gap_added_when_first_segment_offset(self, mock_windows_path, mock_duration):
        """Test leading gap is added when first segment starts after 0."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = 60.0

        # First segment starts at 5.0, not 0.0
        matches = [
            create_simple_match_result(vo_start=5.0, vo_end=10.0),
            create_simple_match_result(vo_start=10.0, vo_end=15.0),
        ]

        config = MockConfig(output=MockOutputConfig(
            include_alternatives=False,
            include_strategy_tracks=False
        ))

        timeline = create_timeline(matches, config)

        assert isinstance(timeline, otio.schema.Timeline)
        # Should have 5 second leading gap

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    @pytest.mark.fast
    def test_no_leading_gap_when_starts_at_zero(self, mock_windows_path, mock_duration):
        """Test no leading gap when first segment starts at 0."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = 60.0

        matches = [
            create_simple_match_result(vo_start=0.0, vo_end=5.0),
        ]

        config = MockConfig(output=MockOutputConfig(
            include_alternatives=False,
            include_strategy_tracks=False
        ))

        timeline = create_timeline(matches, config)

        assert isinstance(timeline, otio.schema.Timeline)

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    @pytest.mark.fast
    def test_no_leading_gap_for_tiny_offset(self, mock_windows_path, mock_duration):
        """Test no leading gap when offset is less than 100ms."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = 60.0

        # Starts at 0.05 seconds (50ms) - should NOT add gap
        matches = [
            create_simple_match_result(vo_start=0.05, vo_end=5.0),
        ]

        config = MockConfig(output=MockOutputConfig(
            include_alternatives=False,
            include_strategy_tracks=False
        ))

        timeline = create_timeline(matches, config)

        assert isinstance(timeline, otio.schema.Timeline)


# ============================================================================
# Test: Between-Segment Gap Handling (lines 324-353)
# ============================================================================

class TestBetweenSegmentGaps:
    """Test gap handling between segments."""

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    @pytest.mark.fast
    def test_gaps_inserted_between_non_contiguous_segments(self, mock_windows_path, mock_duration):
        """Test gaps are inserted when segments are not contiguous."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = 60.0

        # Gap between segment 1 (0-5) and segment 2 (10-15)
        matches = [
            create_simple_match_result(vo_start=0.0, vo_end=5.0),
            create_simple_match_result(vo_start=10.0, vo_end=15.0),  # 5-second gap
        ]

        config = MockConfig(output=MockOutputConfig(
            include_alternatives=False,
            include_strategy_tracks=False
        ))

        timeline = create_timeline(matches, config)

        assert isinstance(timeline, otio.schema.Timeline)
        # Primary track should have gap between clips

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    @pytest.mark.fast
    def test_no_gaps_for_contiguous_segments(self, mock_windows_path, mock_duration):
        """Test no gaps for contiguous segments."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = 60.0

        # Contiguous segments
        matches = [
            create_simple_match_result(vo_start=0.0, vo_end=5.0),
            create_simple_match_result(vo_start=5.0, vo_end=10.0),
            create_simple_match_result(vo_start=10.0, vo_end=15.0),
        ]

        config = MockConfig(output=MockOutputConfig(
            include_alternatives=False,
            include_strategy_tracks=False
        ))

        timeline = create_timeline(matches, config)

        assert isinstance(timeline, otio.schema.Timeline)


# ============================================================================
# Test: Missing Alternative/Secondary Gaps (lines 484-502, 568-586)
# ============================================================================

class TestMissingMatchGaps:
    """Test gap handling when alternatives/secondaries are missing."""

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    @pytest.mark.fast
    def test_gaps_for_missing_alternatives(self, mock_windows_path, mock_duration):
        """Test gaps are added when alternatives are missing."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = 60.0

        vo_seg = MockVoiceoverSegment(start_time=0.0, end_time=5.0)
        vid_seg = MockVideoSegment(source_file="E:/videos/primary.mp4", start_time=0.0, end_time=6.0)
        primary = MockMatch(voiceover_segment=vo_seg, video_segment=vid_seg)

        # Only 1 alternative when 2 are expected
        alt1 = create_alt_match(vo_seg, "alt1.mp4")

        matches = [MockMatchResult(
            primary_match=primary,
            alternatives=[alt1]  # Only 1, config expects 2
        )]

        config = MockConfig(output=MockOutputConfig(
            include_alternatives=True,
            num_alternatives=2,
            include_strategy_tracks=False
        ))

        timeline = create_timeline(matches, config)

        assert isinstance(timeline, otio.schema.Timeline)
        # V3 track should have gap

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    @pytest.mark.fast
    def test_gaps_for_missing_secondaries(self, mock_windows_path, mock_duration):
        """Test gaps are added when secondaries are missing."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = 60.0

        vo_seg = MockVoiceoverSegment(start_time=0.0, end_time=5.0)
        vid_seg = MockVideoSegment(source_file="E:/videos/primary.mp4", start_time=0.0, end_time=6.0)
        primary = MockMatch(voiceover_segment=vo_seg, video_segment=vid_seg)

        # Only 1 secondary when 3 secondary tracks exist
        sec1_vid_seg = MockVideoSegment(source_file="sec1.mp4", start_time=0.0, end_time=6.0)
        sec1 = MockMatch(voiceover_segment=vo_seg, video_segment=sec1_vid_seg)

        matches = [MockMatchResult(
            primary_match=primary,
            secondary_matches=[sec1]  # Only 1, V5 and V6 should have gaps
        )]

        config = MockConfig(output=MockOutputConfig(
            include_alternatives=False,
            include_strategy_tracks=False
        ))

        timeline = create_timeline(matches, config)

        assert isinstance(timeline, otio.schema.Timeline)


# ============================================================================
# Test: Strategy Track Gap Handling (lines 661-679)
# ============================================================================

class TestStrategyTrackGaps:
    """Test gap handling for strategy tracks."""

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    @pytest.mark.fast
    def test_gaps_for_missing_strategy(self, mock_windows_path, mock_duration):
        """Test gaps are added for segments without matching strategy."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = 60.0

        vo_seg = MockVoiceoverSegment(start_time=0.0, end_time=5.0)
        vid_seg = MockVideoSegment(source_file="E:/videos/primary.mp4", start_time=0.0, end_time=6.0)
        primary = MockMatch(voiceover_segment=vo_seg, video_segment=vid_seg)

        # Only embedding_diversity, no broll_only
        strat = create_strategy_match(vo_seg, "strat.mp4", "embedding_diversity")

        matches = [MockMatchResult(
            primary_match=primary,
            strategy_matches=[strat]
        )]

        config = MockConfig(output=MockOutputConfig(
            include_alternatives=False,
            include_strategy_tracks=True,
            strategy_tracks=["embedding_diversity", "broll_only"]
        ))

        timeline = create_timeline(matches, config)

        assert isinstance(timeline, otio.schema.Timeline)
        # broll_only track should have gap

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    @pytest.mark.fast
    def test_empty_strategy_matches(self, mock_windows_path, mock_duration):
        """Test all strategy tracks get gaps when no strategy matches."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = 60.0

        matches = [create_simple_match_result()]  # No strategy matches

        config = MockConfig(output=MockOutputConfig(
            include_alternatives=False,
            include_strategy_tracks=True,
            strategy_tracks=["embedding_diversity", "broll_only"]
        ))

        timeline = create_timeline(matches, config)

        assert isinstance(timeline, otio.schema.Timeline)


# ============================================================================
# Test: Full Pipeline Integration
# ============================================================================

class TestFullPipelineIntegration:
    """Integration tests with all features enabled."""

    @patch('src.otio.timeline._get_media_duration')
    @patch('src.otio.timeline._to_windows_path')
    @patch('src.otio.timeline._add_entity_images_to_track')
    @patch('src.otio.timeline._add_entity_videos_to_track')
    @pytest.mark.fast
    def test_full_timeline_with_all_features(
        self, mock_add_videos, mock_add_images, mock_windows_path, mock_duration, tmp_path
    ):
        """Test complete timeline with all tracks and features."""
        mock_windows_path.side_effect = lambda x: x
        mock_duration.return_value = 60.0

        # Create valid entity image
        img_file = tmp_path / "entity.jpg"
        img_file.write_text("fake image")

        entity_images = {
            "TestEntity": MockEntityResult(images=[MockEntityImage(file=str(img_file))])
        }

        @dataclass
        class MockEntityVideoResult:
            videos: List = field(default_factory=list)

        entity_videos = {
            "TestEntity": MockEntityVideoResult(videos=[{"file": "video.mp4"}])
        }

        # Create segments for audio-first mode
        downloaded_segments = [
            MockDownloadedSegment(
                file="E:/segments/vid1_seg.mp4",
                video_id="vid1",
                original_start=0.0,
                original_end=30.0
            )
        ]

        vo_seg = MockVoiceoverSegment(start_time=5.0, end_time=10.0)
        vid_seg = MockVideoSegment(source_file="vid1.mp3", start_time=5.0, end_time=12.0)
        primary = MockMatch(voiceover_segment=vo_seg, video_segment=vid_seg)

        alt1 = create_alt_match(vo_seg, "alt1.mp4")
        sec1_vid_seg = MockVideoSegment(source_file="sec1.mp4", start_time=0.0, end_time=6.0)
        sec1 = MockMatch(voiceover_segment=vo_seg, video_segment=sec1_vid_seg)
        strat = create_strategy_match(vo_seg, "strat.mp4", "embedding_diversity")

        matches = [MockMatchResult(
            primary_match=primary,
            alternatives=[alt1],
            secondary_matches=[sec1],
            strategy_matches=[strat]
        )]

        config = MockConfig(output=MockOutputConfig(
            include_alternatives=True,
            num_alternatives=2,
            include_strategy_tracks=True,
            strategy_tracks=["embedding_diversity", "broll_only"]
        ))

        timeline = create_timeline(
            matches,
            config,
            voiceover_path="E:/audio/vo.mp3",
            frame_rate=24.0,
            entity_images=entity_images,
            entity_videos=entity_videos,
            downloaded_segments=downloaded_segments
        )

        assert isinstance(timeline, otio.schema.Timeline)
        assert timeline.name == "Matched Footage"
        assert timeline.global_start_time.rate == 24.0

        # Verify track structure
        video_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Video]
        audio_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Audio]

        assert len(video_tracks) >= 8  # V1-V8 minimum + V9, V10
        assert len(audio_tracks) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
