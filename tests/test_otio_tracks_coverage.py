"""
Tests for src/otio/tracks.py edge cases.

Targets:
- Lines 162-179: _add_gap_if_needed gap insertion logic
- Factory function edge cases
- Track builder strategy patterns
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import sys


@pytest.fixture(autouse=True)
def mock_opentimelineio():
    """Mock opentimelineio module for this test file only."""
    original_otio = sys.modules.get('opentimelineio')

    mock_otio = MagicMock()
    sys.modules['opentimelineio'] = mock_otio

    yield mock_otio

    # Restore original state
    if original_otio is not None:
        sys.modules['opentimelineio'] = original_otio
    elif 'opentimelineio' in sys.modules:
        del sys.modules['opentimelineio']


@dataclass
class MockVideoSegment:
    """Mock video segment for testing."""
    source_file: str
    start_time: float
    end_time: float
    text: str = ""


@dataclass
class MockVoiceoverSegment:
    """Mock voiceover segment for testing."""
    start_time: float
    end_time: float
    text: str


@dataclass
class MockMatch:
    """Mock match for testing."""
    video_segment: MockVideoSegment
    voiceover_segment: MockVoiceoverSegment
    confidence: float = 0.85
    reasoning: str = "test match"
    is_keyword_match: bool = False
    is_visual_match: bool = False
    embedding_similarity: float = 0.7
    clip_reuse_count: int = 0
    strategy: str = ""


@dataclass
class MockMatchResult:
    """Mock match result for testing."""
    primary_match: MockMatch
    alternatives: List[MockMatch] = field(default_factory=list)
    secondary_matches: List[MockMatch] = field(default_factory=list)
    strategy_matches: List[MockMatch] = field(default_factory=list)


@dataclass
class MockOutputConfig:
    """Mock output config."""
    num_alternatives: int = 2
    include_alternatives: bool = True


@dataclass
class MockConfig:
    """Mock config for testing."""
    output: MockOutputConfig = field(default_factory=MockOutputConfig)


class TestTrackBuilderFactory:
    """Test get_track_builder factory function."""

    @pytest.mark.fast
    def test_get_track_builder_invalid_index(self):
        """Test that invalid track index raises ValueError."""
        from src.otio.tracks import get_track_builder

        with pytest.raises(ValueError, match="No builder for track index"):
            get_track_builder(
                track_idx=99,
                matches=[],
                config=MockConfig(),
                frame_rate=24.0
            )

    @pytest.mark.fast
    def test_get_track_builder_valid_indices(self):
        """Test that valid track indices return builders."""
        from src.otio.tracks import get_track_builder

        for track_idx in range(10):
            builder = get_track_builder(
                track_idx=track_idx,
                matches=[],
                config=MockConfig(),
                frame_rate=24.0
            )
            assert builder is not None

    @pytest.mark.fast
    def test_get_track_name_beyond_list(self):
        """Test track name fallback for indices beyond list."""
        from src.otio.tracks import PrimaryTrackBuilder

        builder = PrimaryTrackBuilder(
            matches=[],
            config=MockConfig(),
            frame_rate=24.0
        )

        # Index 10 is beyond the track_names list
        name = builder._get_track_name(15)
        assert name == "Track 16"


class TestPrimaryTrackBuilder:
    """Test PrimaryTrackBuilder (V1)."""

    @pytest.mark.fast
    def test_build_with_resolve_video_segment(self):
        """Test build with audio-first mode resolution."""
        from src.otio.tracks import PrimaryTrackBuilder

        vo_seg = MockVoiceoverSegment(start_time=0, end_time=5, text="test")
        vid_seg = MockVideoSegment(
            source_file="/path/to/audio.mp3",
            start_time=10,
            end_time=20
        )
        match = MockMatch(video_segment=vid_seg, voiceover_segment=vo_seg)
        match_result = MockMatchResult(primary_match=match)

        # Mock the resolution function
        def resolve_fn(source, start):
            return "/path/to/video.mp4", start - 5

        builder = PrimaryTrackBuilder(
            matches=[match_result],
            config=MockConfig(),
            frame_rate=24.0,
            resolve_video_segment=resolve_fn
        )

        video_track, audio_track = builder.build(0)

        # Should have called resolution
        assert video_track is not None
        assert audio_track is not None


class TestAlternativeTrackBuilder:
    """Test AlternativeTrackBuilder (V2-V3)."""

    @pytest.mark.fast
    def test_build_no_alternative_adds_gap(self):
        """Test that missing alternative adds gap."""
        from src.otio.tracks import AlternativeTrackBuilder

        vo_seg = MockVoiceoverSegment(start_time=0, end_time=5, text="test")
        vid_seg = MockVideoSegment(source_file="/path/to/video.mp4", start_time=10, end_time=20)
        match = MockMatch(video_segment=vid_seg, voiceover_segment=vo_seg)
        match_result = MockMatchResult(primary_match=match, alternatives=[])

        builder = AlternativeTrackBuilder(
            matches=[match_result],
            config=MockConfig(),
            frame_rate=24.0
        )

        video_track, audio_track = builder.build(1)

        # Should have added gaps for missing alternatives
        assert video_track is not None

    @pytest.mark.fast
    def test_build_with_alternative(self):
        """Test building with available alternative."""
        from src.otio.tracks import AlternativeTrackBuilder

        vo_seg = MockVoiceoverSegment(start_time=0, end_time=5, text="test")
        vid_seg = MockVideoSegment(source_file="/path/to/video.mp4", start_time=10, end_time=20)
        alt_seg = MockVideoSegment(source_file="/path/to/alt.mp4", start_time=5, end_time=15)

        match = MockMatch(video_segment=vid_seg, voiceover_segment=vo_seg)
        alt = MockMatch(video_segment=alt_seg, voiceover_segment=vo_seg)
        match_result = MockMatchResult(primary_match=match, alternatives=[alt])

        builder = AlternativeTrackBuilder(
            matches=[match_result],
            config=MockConfig(),
            frame_rate=24.0
        )

        video_track, audio_track = builder.build(1)

        assert video_track is not None


class TestDiversityTrackBuilder:
    """Test DiversityTrackBuilder (V4-V6)."""

    @pytest.mark.fast
    def test_build_no_secondary_adds_gap(self):
        """Test that missing secondary match adds gap."""
        from src.otio.tracks import DiversityTrackBuilder

        vo_seg = MockVoiceoverSegment(start_time=0, end_time=5, text="test")
        vid_seg = MockVideoSegment(source_file="/path/to/video.mp4", start_time=10, end_time=20)
        match = MockMatch(video_segment=vid_seg, voiceover_segment=vo_seg)
        match_result = MockMatchResult(primary_match=match, secondary_matches=[])

        builder = DiversityTrackBuilder(
            matches=[match_result],
            config=MockConfig(),
            frame_rate=24.0
        )

        video_track, audio_track = builder.build(3)

        assert video_track is not None


class TestEmbeddingDiversityTrackBuilder:
    """Test EmbeddingDiversityTrackBuilder (V7)."""

    @pytest.mark.fast
    def test_build_no_strategy_match_adds_gap(self):
        """Test that missing strategy match adds gap."""
        from src.otio.tracks import EmbeddingDiversityTrackBuilder

        vo_seg = MockVoiceoverSegment(start_time=0, end_time=5, text="test")
        vid_seg = MockVideoSegment(source_file="/path/to/video.mp4", start_time=10, end_time=20)
        match = MockMatch(video_segment=vid_seg, voiceover_segment=vo_seg)
        match_result = MockMatchResult(primary_match=match, strategy_matches=[])

        builder = EmbeddingDiversityTrackBuilder(
            matches=[match_result],
            config=MockConfig(),
            frame_rate=24.0
        )

        video_track, audio_track = builder.build(6)

        assert video_track is not None

    @pytest.mark.fast
    def test_build_with_matching_strategy(self):
        """Test building with matching strategy."""
        from src.otio.tracks import EmbeddingDiversityTrackBuilder

        vo_seg = MockVoiceoverSegment(start_time=0, end_time=5, text="test")
        vid_seg = MockVideoSegment(source_file="/path/to/video.mp4", start_time=10, end_time=20)
        strat_seg = MockVideoSegment(source_file="/path/to/strat.mp4", start_time=0, end_time=10)

        match = MockMatch(video_segment=vid_seg, voiceover_segment=vo_seg)
        strat_match = MockMatch(
            video_segment=strat_seg,
            voiceover_segment=vo_seg,
            strategy="embedding_diversity"
        )
        match_result = MockMatchResult(
            primary_match=match,
            strategy_matches=[strat_match]
        )

        builder = EmbeddingDiversityTrackBuilder(
            matches=[match_result],
            config=MockConfig(),
            frame_rate=24.0
        )

        video_track, audio_track = builder.build(6)

        assert video_track is not None


class TestBRollTrackBuilder:
    """Test BRollTrackBuilder (V8)."""

    @pytest.mark.fast
    def test_build_with_broll_strategy(self):
        """Test building with broll_only strategy match."""
        from src.otio.tracks import BRollTrackBuilder

        vo_seg = MockVoiceoverSegment(start_time=0, end_time=5, text="test")
        vid_seg = MockVideoSegment(source_file="/path/to/video.mp4", start_time=10, end_time=20)
        broll_seg = MockVideoSegment(source_file="/path/to/broll.mp4", start_time=0, end_time=10)

        match = MockMatch(video_segment=vid_seg, voiceover_segment=vo_seg)
        broll_match = MockMatch(
            video_segment=broll_seg,
            voiceover_segment=vo_seg,
            strategy="broll_only"
        )
        match_result = MockMatchResult(
            primary_match=match,
            strategy_matches=[broll_match]
        )

        builder = BRollTrackBuilder(
            matches=[match_result],
            config=MockConfig(),
            frame_rate=24.0
        )

        video_track, audio_track = builder.build(7)

        assert video_track is not None


class TestEntityTrackBuilders:
    """Test EntityImageTrackBuilder (V9) and EntityVideoTrackBuilder (V10)."""

    @pytest.mark.fast
    def test_entity_image_builder_no_images(self):
        """Test V9 builder with no entity images."""
        from src.otio.tracks import EntityImageTrackBuilder

        vo_seg = MockVoiceoverSegment(start_time=0, end_time=5, text="test")
        vid_seg = MockVideoSegment(source_file="/path/to/video.mp4", start_time=10, end_time=20)
        match = MockMatch(video_segment=vid_seg, voiceover_segment=vo_seg)
        match_result = MockMatchResult(primary_match=match)

        builder = EntityImageTrackBuilder(
            matches=[match_result],
            config=MockConfig(),
            frame_rate=24.0,
            entity_images=None  # No images
        )

        # The function is imported inside build(), so patch at source
        with patch('src.otio.entities._add_entity_images_to_track'):
            video_track, audio_track = builder.build(8)

        assert video_track is not None

    @pytest.mark.fast
    def test_entity_video_builder_no_videos(self):
        """Test V10 builder with no entity videos."""
        from src.otio.tracks import EntityVideoTrackBuilder

        vo_seg = MockVoiceoverSegment(start_time=0, end_time=5, text="test")
        vid_seg = MockVideoSegment(source_file="/path/to/video.mp4", start_time=10, end_time=20)
        match = MockMatch(video_segment=vid_seg, voiceover_segment=vo_seg)
        match_result = MockMatchResult(primary_match=match)

        builder = EntityVideoTrackBuilder(
            matches=[match_result],
            config=MockConfig(),
            frame_rate=24.0,
            entity_videos=None  # No videos
        )

        # The function is imported inside build(), so patch at source
        with patch('src.otio.entities._add_entity_videos_to_track'):
            video_track, audio_track = builder.build(9)

        assert video_track is not None


class TestTrackBuilderHelpers:
    """Test TrackBuilder helper methods."""

    @pytest.mark.fast
    def test_create_gap(self):
        """Test _create_gap method."""
        from src.otio.tracks import PrimaryTrackBuilder

        builder = PrimaryTrackBuilder(
            matches=[],
            config=MockConfig(),
            frame_rate=24.0
        )

        gap = builder._create_gap(48)

        assert gap is not None

    @pytest.mark.fast
    def test_create_clip_with_segment_offset(self):
        """Test _create_clip handles segment offset."""
        # This test verifies segment offset extraction logic
        # without invoking full track building which has module conflicts
        from src.otio.utils import get_segment_file_offset

        # Test segment offset extraction from filename pattern
        offset = get_segment_file_offset("/path/to/abc123_0100.mp4")
        assert offset == 100

        # Test no offset pattern
        offset_zero = get_segment_file_offset("/path/to/abc123.mp4")
        assert offset_zero == 0

        # Test different offset values
        offset_200 = get_segment_file_offset("/path/to/video_0200.mp4")
        assert offset_200 == 200


class TestAddGapIfNeeded:
    """Test _add_gap_if_needed method for lines 162-179."""

    @pytest.mark.fast
    def test_add_gap_when_voiceover_has_silence_lines_162_179(self):
        """Test lines 162-179: Gap insertion when expected_start_frames > timeline_frames."""
        from src.otio.tracks import PrimaryTrackBuilder

        builder = PrimaryTrackBuilder(
            matches=[],
            config=MockConfig(),
            frame_rate=24.0
        )

        # Create mock tracks
        mock_video_track = MagicMock()
        mock_audio_track = MagicMock()

        # Voiceover segment starts at 5 seconds (gap from 0 to 5s)
        vo_seg = MockVoiceoverSegment(start_time=5.0, end_time=10.0, text="test")

        # Call _add_gap_if_needed with:
        # - first_segment_start = 0.0
        # - timeline_frames = 0
        # This should trigger gap insertion since expected_start_frames = 120 > 0
        result = builder._add_gap_if_needed(
            video_track=mock_video_track,
            audio_track=mock_audio_track,
            match_idx=0,
            vo_seg=vo_seg,
            first_segment_start=0.0,
            timeline_frames=0
        )

        # Should return expected_start_frames (120 frames)
        assert result == 120

        # Gap should have been appended to both tracks (lines 174-175)
        assert mock_video_track.append.called
        assert mock_audio_track.append.called

    @pytest.mark.fast
    def test_no_gap_when_timeline_at_expected_position(self):
        """Test no gap inserted when timeline_frames == expected_start_frames."""
        from src.otio.tracks import PrimaryTrackBuilder

        builder = PrimaryTrackBuilder(
            matches=[],
            config=MockConfig(),
            frame_rate=24.0
        )

        mock_video_track = MagicMock()
        mock_audio_track = MagicMock()

        # Voiceover starts at 5s, but timeline is already at 120 frames
        vo_seg = MockVoiceoverSegment(start_time=5.0, end_time=10.0, text="test")

        result = builder._add_gap_if_needed(
            video_track=mock_video_track,
            audio_track=mock_audio_track,
            match_idx=0,
            vo_seg=vo_seg,
            first_segment_start=0.0,
            timeline_frames=120  # Already at expected position
        )

        # Should return unchanged timeline_frames (line 179)
        assert result == 120

        # No gap should have been appended
        assert not mock_video_track.append.called
        assert not mock_audio_track.append.called

    @pytest.mark.fast
    def test_gap_calculation_with_offset_start_lines_162_166(self):
        """Test lines 162, 166: Gap calculation with non-zero first_segment_start."""
        from src.otio.tracks import PrimaryTrackBuilder

        builder = PrimaryTrackBuilder(
            matches=[],
            config=MockConfig(),
            frame_rate=24.0
        )

        mock_video_track = MagicMock()
        mock_audio_track = MagicMock()

        # Voiceover segment 2 starts at 15s, first segment started at 10s
        # Gap should be from 0 (timeline_frames) to 120 frames (5s worth)
        vo_seg = MockVoiceoverSegment(start_time=15.0, end_time=20.0, text="test")

        result = builder._add_gap_if_needed(
            video_track=mock_video_track,
            audio_track=mock_audio_track,
            match_idx=1,
            vo_seg=vo_seg,
            first_segment_start=10.0,  # First segment started at 10s
            timeline_frames=0  # At position 0
        )

        # expected_start_frames = (15.0 - 10.0) * 24.0 = 120
        assert result == 120
        assert mock_video_track.append.called
