"""
Tests for _remap_matches_to_video_segments in src/stages/download.py.

This covers lines 535-682 which handle remapping audio file matches
to video segment files after audio-first mode download.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stages.download import DownloadVideoSegmentsStage
from src.state import PipelineState, Match, DownloadedVideo, AudioDownload


# ============================================================================
# Mock Classes
# ============================================================================

@dataclass
class MockDownloadedSegment:
    """Mock DownloadedSegment from downloader/types.py"""
    video_id: str
    file: str
    original_start: float
    matches: list


@dataclass
class MockVideoSegment:
    """Mock video segment with source_file"""
    source_file: str
    start_time: float
    end_time: float
    text: str = ""


class MockMatchResult:
    """Mock MatchResult with primary_match, alternatives, etc."""
    def __init__(self, primary_match):
        self.primary_match = primary_match
        self.alternatives = []
        self.secondary_matches = []
        self.strategy_matches = []


class MockAlternativeMatch:
    """Mock AlternativeMatch with video_segment"""
    def __init__(self, video_segment):
        self.video_segment = video_segment


# ============================================================================
# Test _remap_matches_to_video_segments
# ============================================================================

class TestRemapMatchesToVideoSegments:
    """Test _remap_matches_to_video_segments method"""

    @pytest.fixture
    def stage(self):
        return DownloadVideoSegmentsStage()

    @pytest.mark.fast
    def test_remap_basic_match(self, stage):
        """Test remapping a basic Match object"""
        state = PipelineState()

        # Create match pointing to audio file
        match = Match(
            segment_index=0,
            video_file="/path/to/audio_abc123.mp3",
            video_start=10.0,
            video_end=20.0,
            confidence=0.9
        )
        state.matches = [match]

        # Create audio downloads
        audio_downloads_by_id = {
            "abc123": Mock(file="/path/to/audio_abc123.mp3")
        }

        # Create downloaded segments
        downloaded_segments = [
            MockDownloadedSegment(
                video_id="abc123",
                file="/path/to/video_abc123_0010.mp4",
                original_start=10.0,
                matches=[Mock(start_time=10.0)]
            )
        ]

        stage._remap_matches_to_video_segments(state, downloaded_segments, audio_downloads_by_id)

        assert match.video_file == "/path/to/video_abc123_0010.mp4"

    @pytest.mark.fast
    def test_remap_skips_stock_videos(self, stage):
        """Test remapping skips stock video files (pexels_, pixabay_)"""
        state = PipelineState()

        # Create match pointing to stock video
        pexels_match = Match(
            segment_index=0,
            video_file="/path/to/pexels_beach.mp4",
            video_start=0.0,
            video_end=5.0,
            confidence=0.8
        )
        pixabay_match = Match(
            segment_index=1,
            video_file="/path/to/pixabay_nature.mp4",
            video_start=0.0,
            video_end=5.0,
            confidence=0.7
        )
        state.matches = [pexels_match, pixabay_match]

        stage._remap_matches_to_video_segments(state, [], {})

        # Should not be modified
        assert "pexels_beach" in pexels_match.video_file
        assert "pixabay_nature" in pixabay_match.video_file

    @pytest.mark.fast
    def test_remap_skips_entity_videos(self, stage):
        """Test remapping skips entity videos"""
        state = PipelineState()

        match = Match(
            segment_index=0,
            video_file="/path/to/entity_tower.mp4",
            video_start=0.0,
            video_end=5.0,
            confidence=0.8
        )
        state.matches = [match]

        stage._remap_matches_to_video_segments(state, [], {})

        # Should not be modified
        assert "entity_tower" in match.video_file

    @pytest.mark.fast
    def test_remap_handles_match_result_object(self, stage):
        """Test remapping MatchResult objects with primary_match"""
        state = PipelineState()

        primary = Match(
            segment_index=0,
            video_file="/path/to/audio_def456.mp3",
            video_start=20.0,
            video_end=30.0,
            confidence=0.9
        )
        match_result = MockMatchResult(primary)
        state.matches = [match_result]

        audio_downloads_by_id = {
            "def456": Mock(file="/path/to/audio_def456.mp3")
        }

        downloaded_segments = [
            MockDownloadedSegment(
                video_id="def456",
                file="/path/to/video_def456_0020.mp4",
                original_start=20.0,
                matches=[Mock(start_time=20.0)]
            )
        ]

        stage._remap_matches_to_video_segments(state, downloaded_segments, audio_downloads_by_id)

        assert primary.video_file == "/path/to/video_def456_0020.mp4"

    @pytest.mark.fast
    def test_remap_match_result_alternatives(self, stage):
        """Test remapping MatchResult alternatives"""
        state = PipelineState()

        primary = Match(
            segment_index=0,
            video_file="/path/to/pexels_skip.mp4",  # Stock video, skip
            video_start=0.0,
            video_end=5.0,
            confidence=0.9
        )

        alt_segment = MockVideoSegment(
            source_file="/path/to/audio_ghi789.mp3",
            start_time=30.0,
            end_time=40.0
        )
        alt_match = MockAlternativeMatch(alt_segment)

        match_result = MockMatchResult(primary)
        match_result.alternatives = [alt_match]
        state.matches = [match_result]

        audio_downloads_by_id = {
            "ghi789": Mock(file="/path/to/audio_ghi789.mp3")
        }

        downloaded_segments = [
            MockDownloadedSegment(
                video_id="ghi789",
                file="/path/to/video_ghi789_0030.mp4",
                original_start=30.0,
                matches=[Mock(start_time=30.0)]
            )
        ]

        stage._remap_matches_to_video_segments(state, downloaded_segments, audio_downloads_by_id)

        assert alt_segment.source_file == "/path/to/video_ghi789_0030.mp4"

    @pytest.mark.fast
    def test_remap_match_result_secondary_matches(self, stage):
        """Test remapping MatchResult secondary matches"""
        state = PipelineState()

        primary = Match(
            segment_index=0,
            video_file="/path/to/pexels_skip.mp4",
            video_start=0.0,
            video_end=5.0,
            confidence=0.9
        )

        sec_segment = MockVideoSegment(
            source_file="/path/to/audio_jkl012.mp3",
            start_time=45.0,
            end_time=55.0
        )
        sec_match = MockAlternativeMatch(sec_segment)

        match_result = MockMatchResult(primary)
        match_result.secondary_matches = [sec_match]
        state.matches = [match_result]

        audio_downloads_by_id = {
            "jkl012": Mock(file="/path/to/audio_jkl012.mp3")
        }

        downloaded_segments = [
            MockDownloadedSegment(
                video_id="jkl012",
                file="/path/to/video_jkl012_0045.mp4",
                original_start=45.0,
                matches=[Mock(start_time=45.0)]
            )
        ]

        stage._remap_matches_to_video_segments(state, downloaded_segments, audio_downloads_by_id)

        assert sec_segment.source_file == "/path/to/video_jkl012_0045.mp4"

    @pytest.mark.fast
    def test_remap_match_result_strategy_matches(self, stage):
        """Test remapping MatchResult strategy matches"""
        state = PipelineState()

        primary = Match(
            segment_index=0,
            video_file="/path/to/entity_skip.mp4",
            video_start=0.0,
            video_end=5.0,
            confidence=0.9
        )

        strat_segment = MockVideoSegment(
            source_file="/path/to/audio_mno345.mp3",
            start_time=60.0,
            end_time=70.0
        )
        strat_match = MockAlternativeMatch(strat_segment)

        match_result = MockMatchResult(primary)
        match_result.strategy_matches = [strat_match]
        state.matches = [match_result]

        audio_downloads_by_id = {
            "mno345": Mock(file="/path/to/audio_mno345.mp3")
        }

        downloaded_segments = [
            MockDownloadedSegment(
                video_id="mno345",
                file="/path/to/video_mno345_0060.mp4",
                original_start=60.0,
                matches=[Mock(start_time=60.0)]
            )
        ]

        stage._remap_matches_to_video_segments(state, downloaded_segments, audio_downloads_by_id)

        assert strat_segment.source_file == "/path/to/video_mno345_0060.mp4"

    @pytest.mark.fast
    def test_remap_handles_video_segment_match(self, stage):
        """Test remapping Match with video_segment attribute"""
        state = PipelineState()

        # Create match with video_segment (utils.Match style)
        # Use spec=[] to prevent auto-attribute creation, then add video_segment
        video_segment = MockVideoSegment(
            source_file="/path/to/audio_pqr678.mp3",
            start_time=75.0,
            end_time=85.0
        )
        match = Mock(spec=['video_segment'])
        match.video_segment = video_segment

        state.matches = [match]

        audio_downloads_by_id = {
            "pqr678": Mock(file="/path/to/audio_pqr678.mp3")
        }

        downloaded_segments = [
            MockDownloadedSegment(
                video_id="pqr678",
                file="/path/to/video_pqr678_0075.mp4",
                original_start=75.0,
                matches=[Mock(start_time=75.0)]
            )
        ]

        stage._remap_matches_to_video_segments(state, downloaded_segments, audio_downloads_by_id)

        assert video_segment.source_file == "/path/to/video_pqr678_0075.mp4"

    @pytest.mark.fast
    def test_remap_warns_unknown_match_structure(self, stage):
        """Test logs warning for unknown Match structure"""
        state = PipelineState()

        # Create match without video_file or video_segment
        strange_match = Mock(spec=[])
        state.matches = [strange_match]

        # Should not raise, just warn
        stage._remap_matches_to_video_segments(state, [], {})

    @pytest.mark.fast
    def test_remap_warns_missing_video_id(self, stage):
        """Test logs warning when video_id not found"""
        state = PipelineState()

        match = Match(
            segment_index=0,
            video_file="/path/to/unknown_video.mp3",
            video_start=10.0,
            video_end=20.0,
            confidence=0.9
        )
        state.matches = [match]

        # Empty audio_downloads_by_id
        audio_downloads_by_id = {}

        # Should not raise, just warn
        stage._remap_matches_to_video_segments(state, [], audio_downloads_by_id)

    @pytest.mark.fast
    def test_remap_warns_missing_segment(self, stage):
        """Test logs warning when segment not found for time"""
        state = PipelineState()

        match = Match(
            segment_index=0,
            video_file="/path/to/audio_stu901.mp3",
            video_start=999.0,  # Time not in segments
            video_end=1009.0,
            confidence=0.9
        )
        state.matches = [match]

        audio_downloads_by_id = {
            "stu901": Mock(file="/path/to/audio_stu901.mp3")
        }

        # Segment at different time
        downloaded_segments = [
            MockDownloadedSegment(
                video_id="stu901",
                file="/path/to/video_stu901_0000.mp4",
                original_start=0.0,
                matches=[Mock(start_time=0.0)]
            )
        ]

        stage._remap_matches_to_video_segments(state, downloaded_segments, audio_downloads_by_id)

        # Match should not be updated (no segment at time 999)
        assert "audio_stu901.mp3" in match.video_file

    @pytest.mark.fast
    def test_remap_handles_timestamped_audio_filename(self, stage):
        """Test handles audio filename with timestamp suffix"""
        state = PipelineState()

        # Audio file has timestamp suffix _0045
        match = Match(
            segment_index=0,
            video_file="/path/to/vwx234_0045.mp3",
            video_start=50.0,
            video_end=60.0,
            confidence=0.9
        )
        state.matches = [match]

        audio_downloads_by_id = {
            "vwx234": Mock(file="/path/to/vwx234.mp3")  # Base without timestamp
        }

        downloaded_segments = [
            MockDownloadedSegment(
                video_id="vwx234",
                file="/path/to/video_vwx234_0050.mp4",
                original_start=50.0,
                matches=[Mock(start_time=50.0)]
            )
        ]

        stage._remap_matches_to_video_segments(state, downloaded_segments, audio_downloads_by_id)

        assert match.video_file == "/path/to/video_vwx234_0050.mp4"

    @pytest.mark.fast
    def test_remap_handles_audio_dict_format(self, stage):
        """Test handles audio_downloads_by_id with dict values"""
        state = PipelineState()

        match = Match(
            segment_index=0,
            video_file="/path/to/yz0567.mp3",
            video_start=80.0,
            video_end=90.0,
            confidence=0.9
        )
        state.matches = [match]

        # Dict format instead of object
        audio_downloads_by_id = {
            "yz0567": {"file": "/path/to/yz0567.mp3"}
        }

        downloaded_segments = [
            MockDownloadedSegment(
                video_id="yz0567",
                file="/path/to/video_yz0567_0080.mp4",
                original_start=80.0,
                matches=[Mock(start_time=80.0)]
            )
        ]

        stage._remap_matches_to_video_segments(state, downloaded_segments, audio_downloads_by_id)

        assert match.video_file == "/path/to/video_yz0567_0080.mp4"

    @pytest.mark.fast
    def test_remap_empty_matches(self, stage):
        """Test with empty matches list"""
        state = PipelineState()
        state.matches = []

        # Should not raise
        stage._remap_matches_to_video_segments(state, [], {})

    @pytest.mark.fast
    def test_remap_counts_updates(self, stage):
        """Test counts updated matches correctly"""
        state = PipelineState()

        match1 = Match(
            segment_index=0,
            video_file="/path/to/audio_a1b2c3.mp3",
            video_start=0.0,
            video_end=10.0,
            confidence=0.9
        )
        match2 = Match(
            segment_index=1,
            video_file="/path/to/audio_a1b2c3.mp3",
            video_start=10.0,
            video_end=20.0,
            confidence=0.8
        )
        state.matches = [match1, match2]

        audio_downloads_by_id = {
            "a1b2c3": Mock(file="/path/to/audio_a1b2c3.mp3")
        }

        downloaded_segments = [
            MockDownloadedSegment(
                video_id="a1b2c3",
                file="/path/to/video_a1b2c3_0000.mp4",
                original_start=0.0,
                matches=[Mock(start_time=0.0)]
            ),
            MockDownloadedSegment(
                video_id="a1b2c3",
                file="/path/to/video_a1b2c3_0010.mp4",
                original_start=10.0,
                matches=[Mock(start_time=10.0)]
            )
        ]

        stage._remap_matches_to_video_segments(state, downloaded_segments, audio_downloads_by_id)

        # Both matches should be updated
        assert "video_a1b2c3_0000.mp4" in match1.video_file
        assert "video_a1b2c3_0010.mp4" in match2.video_file


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
