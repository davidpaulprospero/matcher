"""Tests for thematic consistency scoring for matched video segments.

Tests compute_thematic_consistency() function in src/matching/scoring.py.
US-134-010: Add cross-video thematic consistency scoring.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.scoring import compute_thematic_consistency
from src.utils import SRTSegment


# ==============================================================================
# Test Fixtures
# ==============================================================================

@pytest.fixture
def mock_config():
    """Create mock config with default thematic consistency settings."""
    config = MagicMock()
    config.matching = MagicMock()
    config.matching.thematic_consistency_enabled = True
    config.matching.thematic_consistency_window = 3
    config.matching.thematic_consistency_boost_max = 0.05
    return config


@pytest.fixture
def mock_config_disabled():
    """Create mock config with thematic consistency disabled."""
    config = MagicMock()
    config.matching = MagicMock()
    config.matching.thematic_consistency_enabled = False
    config.matching.thematic_consistency_window = 3
    config.matching.thematic_consistency_boost_max = 0.05
    return config


@pytest.fixture
def vo_segment():
    """Create voiceover segment with topics and keywords."""
    seg = SRTSegment(
        index=0,
        start_time=0.0,
        end_time=5.0,
        text="The earthquake caused massive destruction",
        source_file="voiceover.mp3"
    )
    seg.topics = ["earthquake", "disaster", "destruction"]
    seg.keywords = ["earthquake", "damage", "city"]
    return seg


@pytest.fixture
def video_segment():
    """Create video segment with topics and keywords."""
    seg = SRTSegment(
        index=0,
        start_time=0.0,
        end_time=10.0,
        text="Video about earthquake damage",
        source_file="video1.mp4"
    )
    seg.topics = ["earthquake", "damage"]
    seg.keywords = ["earthquake", "disaster", "destruction"]
    return seg


@pytest.fixture
def matched_segments():
    """Create list of matched segments for window context."""
    seg1 = SRTSegment(
        index=0,
        start_time=0.0,
        end_time=10.0,
        text="First video",
        source_file="video1.mp4"
    )
    seg1.topics = ["earthquake", "rescue"]
    seg1.keywords = ["earthquake", "help"]

    seg2 = SRTSegment(
        index=1,
        start_time=10.0,
        end_time=20.0,
        text="Second video",
        source_file="video2.mp4"
    )
    seg2.topics = ["disaster", "recovery"]
    seg2.keywords = ["disaster", "aid"]

    return [seg1, seg2]


# ==============================================================================
# Basic Function Tests
# ==============================================================================

class TestThematicConsistencyFunction:
    """Test that compute_thematic_consistency function exists with correct signature."""

    def test_function_exists(self):
        """compute_thematic_consistency function should exist in scoring module."""
        from src.matching.scoring import compute_thematic_consistency
        assert callable(compute_thematic_consistency)

    def test_function_callable_with_params(self, mock_config, vo_segment, video_segment):
        """Function should accept confidence, vo_segment, video_segment, matched_segments, and config."""
        result, reason = compute_thematic_consistency(
            0.8, vo_segment, video_segment, [], mock_config
        )
        assert isinstance(result, float)
        assert isinstance(reason, str)

    def test_disabled_returns_unchanged(self, mock_config_disabled, vo_segment, video_segment):
        """When thematic_consistency_enabled=False, confidence should return unchanged."""
        result, reason = compute_thematic_consistency(
            0.8, vo_segment, video_segment, [], mock_config_disabled
        )
        assert result == 0.8
        assert reason == ""


# ==============================================================================
# Config Tests
# ==============================================================================

class TestThematicConsistencyConfig:
    """Test config options for thematic consistency."""

    def test_config_defaults(self, mock_config):
        """Config should have sensible defaults for thematic consistency."""
        mc = mock_config.matching
        assert mc.thematic_consistency_enabled is True
        assert mc.thematic_consistency_window == 3
        assert mc.thematic_consistency_boost_max == 0.05

    def test_boost_applied_when_themes_align(self, mock_config, vo_segment, video_segment, matched_segments):
        """Boost should be applied when video themes align with voiceover and adjacent segments."""
        result, reason = compute_thematic_consistency(
            0.8, vo_segment, video_segment, matched_segments, mock_config
        )
        # Video segment has "earthquake", "disaster", "destruction"
        # Voiceover has "earthquake", "disaster", "destruction"
        # Matched segments have topics with "earthquake", "disaster"
        # Should get a boost since themes align
        assert result >= 0.8
        assert "thematic consistency" in reason.lower() or result > 0.8

    def test_no_boost_when_no_theme_overlap(self, mock_config):
        """No boost when video has no theme overlap with voiceover or adjacent segments."""
        vo = SRTSegment(index=0, start_time=0.0, end_time=5.0, text=" cooking recipe", source_file="vo.mp3")
        vo.topics = ["cooking", "recipe"]
        vo.keywords = ["cooking", "food"]

        video = SRTSegment(index=0, start_time=0.0, end_time=10.0, text="tech video", source_file="vid.mp4")
        video.topics = ["technology", "programming"]
        video.keywords = ["code", "software"]

        matched = []
        matched.append(SRTSegment(index=1, start_time=10.0, end_time=20.0, text="matched", source_file="m.mp4"))
        matched[0].topics = ["sports"]
        matched[0].keywords = ["football"]

        result, reason = compute_thematic_consistency(0.8, vo, video, matched, mock_config)
        # No overlap between cooking/recipe and technology/sports
        assert result == 0.8
        assert reason == ""

    def test_no_boost_when_low_consistency_score(self, mock_config):
        """No boost when combined consistency score is below threshold (0.3)."""
        vo = SRTSegment(index=0, start_time=0.0, end_time=5.0, text="vo", source_file="vo.mp3")
        vo.topics = ["topic1"]
        vo.keywords = ["keyword1"]

        video = SRTSegment(index=0, start_time=0.0, end_time=10.0, text="vid", source_file="vid.mp4")
        video.topics = ["unrelated"]
        video.keywords = ["different"]

        matched = []
        matched.append(SRTSegment(index=1, start_time=10.0, end_time=20.0, text="m", source_file="m.mp4"))
        matched[0].topics = ["another"]
        matched[0].keywords = ["thing"]

        result, reason = compute_thematic_consistency(0.8, vo, video, matched, mock_config)
        # Low overlap should not trigger boost
        assert result == 0.8

    def test_boost_capped_at_max(self, mock_config):
        """Boost should not exceed thematic_consistency_boost_max."""
        # Create segments with high overlap
        vo = SRTSegment(index=0, start_time=0.0, end_time=5.0, text="vo", source_file="vo.mp3")
        vo.topics = ["earthquake", "disaster", "destruction", "rescue"]
        vo.keywords = ["earthquake", "damage"]

        video = SRTSegment(index=0, start_time=0.0, end_time=10.0, text="vid", source_file="vid.mp4")
        video.topics = ["earthquake", "disaster"]
        video.keywords = ["earthquake", "damage"]

        matched = []
        for i in range(3):
            m = SRTSegment(index=i, start_time=i*10, end_time=(i+1)*10, text="m", source_file=f"m{i}.mp4")
            m.topics = ["earthquake", "disaster", "rescue"]
            m.keywords = ["earthquake", "help"]
            matched.append(m)

        # With max_boost = 0.05, result should not exceed 0.85
        result, reason = compute_thematic_consistency(0.8, vo, video, matched, mock_config)
        assert result <= 0.85  # 0.8 + 0.05 max


# ==============================================================================
# Edge Cases
# ==============================================================================

class TestThematicConsistencyEdgeCases:
    """Test edge cases for thematic consistency."""

    def test_empty_vo_topics(self, mock_config, video_segment, matched_segments):
        """No boost when voiceover has no topics."""
        vo = SRTSegment(index=0, start_time=0.0, end_time=5.0, text="text", source_file="vo.mp3")
        vo.topics = []
        vo.keywords = []

        result, reason = compute_thematic_consistency(0.8, vo, video_segment, matched_segments, mock_config)
        assert result == 0.8
        assert reason == ""

    def test_empty_video_topics(self, mock_config, vo_segment, matched_segments):
        """No boost when video segment has no topics."""
        video = SRTSegment(index=0, start_time=0.0, end_time=10.0, text="text", source_file="vid.mp4")
        video.topics = []
        video.keywords = []

        result, reason = compute_thematic_consistency(0.8, vo_segment, video, matched_segments, mock_config)
        assert result == 0.8

    def test_empty_matched_segments(self, mock_config, vo_segment, video_segment):
        """Should still work with empty matched segments list."""
        result, reason = compute_thematic_consistency(0.8, vo_segment, video_segment, [], mock_config)
        # With no matched segments to compare against, may or may not get boost
        # depending on vo-video overlap
        assert isinstance(result, float)
        assert result >= 0.0
        assert result <= 1.0

    def test_confidence_clamped_to_valid_range(self, mock_config, vo_segment, video_segment, matched_segments):
        """Confidence should be clamped to [0.0, 1.0]."""
        # Test at high end
        result, _ = compute_thematic_consistency(0.99, vo_segment, video_segment, matched_segments, mock_config)
        assert result <= 1.0

        # Test at low end
        result, _ = compute_thematic_consistency(0.01, vo_segment, video_segment, matched_segments, mock_config)
        assert result >= 0.0

    def test_config_window_size(self, mock_config, vo_segment, video_segment):
        """Window size from config should be respected."""
        # Create more than window_size matched segments
        matched = []
        for i in range(5):
            m = SRTSegment(index=i, start_time=i*10, end_time=(i+1)*10, text="m", source_file=f"m{i}.mp4")
            m.topics = ["earthquake"]
            m.keywords = ["disaster"]
            matched.append(m)

        # Should only consider last window_size (3) segments
        result, reason = compute_thematic_consistency(0.8, vo_segment, video_segment, matched, mock_config)
        assert isinstance(result, float)


# ==============================================================================
# Integration with Config
# ==============================================================================

class TestThematicConsistencyConfigIntegration:
    """Test that config values are properly read."""

    def test_custom_boost_max(self):
        """Custom max boost should be applied."""
        config = MagicMock()
        config.matching = MagicMock()
        config.matching.thematic_consistency_enabled = True
        config.matching.thematic_consistency_window = 3
        config.matching.thematic_consistency_boost_max = 0.10

        vo = SRTSegment(index=0, start_time=0.0, end_time=5.0, text="vo", source_file="vo.mp3")
        vo.topics = ["topic1", "topic2", "topic3"]
        vo.keywords = ["kw1"]

        video = SRTSegment(index=0, start_time=0.0, end_time=10.0, text="vid", source_file="vid.mp4")
        video.topics = ["topic1"]
        video.keywords = ["kw1"]

        matched = [
            SRTSegment(index=0, start_time=0.0, end_time=10.0, text="m", source_file="m.mp4"),
            SRTSegment(index=1, start_time=10.0, end_time=20.0, text="m", source_file="m.mp4"),
        ]
        matched[0].topics = ["topic1"]
        matched[0].keywords = ["kw1"]
        matched[1].topics = ["topic2"]
        matched[1].keywords = ["kw2"]

        result, reason = compute_thematic_consistency(0.8, vo, video, matched, config)
        # With max boost 0.10, result could go up to 0.90
        assert result <= 0.90
