"""Tests for US-129-010: Pre-queue segment validation.

Tests the segment validation logic that checks:
- Segment duration >= min_duration_seconds
- Segment duration <= max_duration_seconds
- Segment start/end within video duration bounds
- Duplicate segments filtered
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any

from src.stages.download_segments import DownloadVideoSegmentsStage
from src.state import PipelineState, Match, VideoSearchResult


class MockVideoSegment:
    """Mock video segment for testing."""
    def __init__(self, source_file: str, start_time: float, end_time: float):
        self.source_file = source_file
        self.start_time = start_time
        self.end_time = end_time


class MockPrimaryMatch:
    """Mock primary match for testing."""
    def __init__(self, video_segment: MockVideoSegment, confidence: float = 0.5):
        self.video_segment = video_segment
        self.confidence = confidence


class MockMatch:
    """Mock match for testing."""
    def __init__(
        self,
        video_id: str,
        start_time: float,
        end_time: float,
        confidence: float = 0.5,
        segment_index: int = 0
    ):
        self.video_id = video_id
        self.segment_index = segment_index
        # Create primary_match structure
        self.primary_match = MockPrimaryMatch(
            video_segment=MockVideoSegment(video_id, start_time, end_time),
            confidence=confidence
        )


class MockDownloadConfig:
    """Mock download config with segment validation."""
    def __init__(self, **kwargs):
        self.segment_buffer = 5.0
        self.segment_validation = MockSegmentValidation(**kwargs)


class MockSegmentValidation:
    """Mock segment validation config."""
    def __init__(
        self,
        min_duration_seconds: float = 1.0,
        max_duration_seconds: float = 0.0,
        validate_bounds: bool = True,
        filter_duplicates: bool = True,
        strictness: str = "lenient",
        log_skipped: bool = True
    ):
        self.min_duration_seconds = min_duration_seconds
        self.max_duration_seconds = max_duration_seconds
        self.validate_bounds = validate_bounds
        self.filter_duplicates = filter_duplicates
        self.strictness = strictness
        self.log_skipped = log_skipped


@pytest.fixture
def stage():
    """Create a DownloadVideoSegmentsStage for testing."""
    return DownloadVideoSegmentsStage()


@pytest.fixture
def state_with_matches():
    """Create a PipelineState with test matches."""
    state = PipelineState()
    # Use segments that meet min_duration of 1.0s
    state.matches = [
        MockMatch("video1", 10.0, 20.0, 0.8, segment_index=0),  # 10s duration
        MockMatch("video2", 30.0, 45.0, 0.7, segment_index=1),  # 15s duration
    ]
    # Add video search results with durations - use dataclass-like objects
    class MockVideoSearchResult:
        def __init__(self, vid, dur):
            self.video_id = vid
            self.duration = dur
    state.video_search_results = [
        MockVideoSearchResult("video1", 300.0),
        MockVideoSearchResult("video2", 600.0),
    ]
    return state


class TestSegmentValidation:
    """Test segment validation logic."""

    def test_validate_duration_too_short(self, stage, state_with_matches):
        """Test that segments shorter than min_duration are skipped."""
        # Create config with min_duration = 15s
        download_config = MockDownloadConfig(min_duration_seconds=15.0)

        # Add a segment that's only 5 seconds
        state_with_matches.matches.append(
            MockMatch("video3", 100.0, 105.0, 0.6)  # Only 5s duration
        )

        with patch('src.stages.download_segments.logger') as mock_logger:
            segments = stage._collect_matched_segments(
                state_with_matches, 5.0, download_config
            )

        # Should have 2 valid segments (video1 has 10s, video2 has 15s - both >= 15s after min check)
        # Wait, video1 is 10s which is < 15s - so it should be filtered too
        # Actually video2 is 15s which is >= 15s, so should pass
        # Let me reconsider - with min=15s, video1 (10s) and video3 (5s) are filtered
        # So we should have 1 segment (video2)
        assert len(segments) == 1
        assert segments[0]['video_id'] == 'video2'

    def test_validate_duration_too_long(self, stage, state_with_matches):
        """Test that segments longer than max_duration are skipped."""
        # Create config with max_duration = 12s
        download_config = MockDownloadConfig(max_duration_seconds=12.0)

        # video1 = 10s (passes), video2 = 15s (fails)
        with patch('src.stages.download_segments.logger') as mock_logger:
            segments = stage._collect_matched_segments(
                state_with_matches, 5.0, download_config
            )

        # Should have 1 valid segment (video1)
        assert len(segments) == 1
        assert segments[0]['video_id'] == 'video1'

    def test_validate_bounds_out_of_bounds(self, stage, state_with_matches):
        """Test that segments outside video duration are skipped."""
        # video1 has duration 300.0
        # video2 has duration 600.0
        # Create config with bounds validation
        download_config = MockDownloadConfig(validate_bounds=True)

        # Add a segment that exceeds video duration for video1
        state_with_matches.matches.append(
            MockMatch("video1", 250.0, 400.0, 0.6)  # Ends at 400s, but video1 is only 300s
        )

        with patch('src.stages.download_segments.logger') as mock_logger:
            segments = stage._collect_matched_segments(
                state_with_matches, 5.0, download_config
            )

        # Should have 3 segments (original 2 + video1 new 400s)
        # But video1 250-400 is out of bounds, so should be filtered
        # Wait - it's being added AFTER the initial 2, so we should have 3 - 1 = 2
        # Actually let me reconsider - the new segment is added to matches
        # Then it gets processed and filtered
        # So we get 2 original + 1 new - 1 filtered = 2
        assert len(segments) == 2

    def test_validate_duplicates_filtered(self, stage, state_with_matches):
        """Test that duplicate segments are filtered."""
        # Add duplicate matches
        state_with_matches.matches.append(
            MockMatch("video1", 10.0, 20.0, 0.8, segment_index=2)  # Duplicate of video1
        )
        state_with_matches.matches.append(
            MockMatch("video2", 30.0, 45.0, 0.7, segment_index=3)  # Duplicate of video2
        )

        download_config = MockDownloadConfig(filter_duplicates=True)

        with patch('src.stages.download_segments.logger') as mock_logger:
            segments = stage._collect_matched_segments(
                state_with_matches, 5.0, download_config
            )

        # Should have 2 unique segments (duplicates filtered)
        assert len(segments) == 2

    def test_validate_no_config(self, stage, state_with_matches):
        """Test that validation is skipped when no config provided."""
        segments = stage._collect_matched_segments(
            state_with_matches, 5.0, None
        )

        # Should have all 2 segments (no validation)
        assert len(segments) == 2

    def test_validate_strict_mode_raises(self, stage, state_with_matches):
        """Test that strict mode raises on invalid segment."""
        download_config = MockDownloadConfig(
            min_duration_seconds=15.0,
            strictness="strict"
        )

        # Add an invalid segment - but video1 is also < 15s now
        # This will cause the first invalid to trigger

        with pytest.raises(ValueError, match="duration.*<.*min"):
            stage._collect_matched_segments(
                state_with_matches, 5.0, download_config
            )

    def test_validate_lenient_mode_continues(self, stage, state_with_matches):
        """Test that lenient mode skips invalid and continues."""
        download_config = MockDownloadConfig(
            min_duration_seconds=15.0,
            strictness="lenient"
        )

        # video1 is 10s (< 15s), video2 is 15s (>= 15s)
        # So video1 should be filtered
        segments = stage._collect_matched_segments(
            state_with_matches, 5.0, download_config
        )

        # Should have 1 valid segment (video2 only)
        assert len(segments) == 1
        assert segments[0]['video_id'] == 'video2'

    def test_validate_min_zero_disables(self, stage, state_with_matches):
        """Test that min_duration=0 disables min duration check."""
        download_config = MockDownloadConfig(min_duration_seconds=0.0)

        # Add a very short segment
        state_with_matches.matches.append(
            MockMatch("video3", 100.0, 100.5, 0.6)  # Only 0.5s
        )

        segments = stage._collect_matched_segments(
            state_with_matches, 5.0, download_config
        )

        # Should have 3 segments (min duration check disabled)
        assert len(segments) == 3


class TestSegmentValidationConfig:
    """Test SegmentValidationConfig class."""

    def test_valid_config(self):
        """Test valid config passes validation."""
        from src.config.sections.download import SegmentValidationConfig

        config = SegmentValidationConfig(
            min_duration_seconds=1.0,
            max_duration_seconds=30.0,
            validate_bounds=True,
            filter_duplicates=True,
            strictness="lenient"
        )

        assert config.min_duration_seconds == 1.0
        assert config.max_duration_seconds == 30.0

    def test_invalid_min_negative(self):
        """Test negative min_duration raises."""
        from src.config.sections.download import SegmentValidationConfig

        with pytest.raises(ValueError, match="min_duration_seconds.*>=.*0"):
            SegmentValidationConfig(min_duration_seconds=-1.0)

    def test_invalid_max_negative(self):
        """Test negative max_duration raises."""
        from src.config.sections.download import SegmentValidationConfig

        with pytest.raises(ValueError, match="max_duration_seconds.*>=.*0"):
            SegmentValidationConfig(max_duration_seconds=-1.0)

    def test_invalid_min_max_order(self):
        """Test min > max raises."""
        from src.config.sections.download import SegmentValidationConfig

        with pytest.raises(ValueError, match="min_duration_seconds.*<=.*max_duration_seconds"):
            SegmentValidationConfig(
                min_duration_seconds=10.0,
                max_duration_seconds=5.0
            )

    def test_invalid_strictness(self):
        """Test invalid strictness raises."""
        from src.config.sections.download import SegmentValidationConfig

        with pytest.raises(ValueError, match="strictness must be one of"):
            SegmentValidationConfig(strictness="invalid")
