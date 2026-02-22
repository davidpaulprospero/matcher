"""Tests for apply_source_channel_consistency (US-95-006).

Tests the standalone apply_source_channel_consistency function that rewards
videos from the same YouTube channel as previous matches.
"""

import pytest
from unittest.mock import Mock, MagicMock
from src.utils import SRTSegment
from src.state import Match


class MockConfig:
    """Mock config for testing."""
    def __init__(self, source_channel_coherence_boost=0.05):
        self.matching = Mock()
        self.matching.source_channel_coherence_boost = source_channel_coherence_boost


class TestUS95SourceChannelConsistency:
    """Tests for apply_source_channel_consistency (US-95-006)."""

    def test_same_channel_applies_boost(self):
        """Same channel as previous match should get a boost."""
        from src.matching.scoring import apply_source_channel_consistency

        vo_segment = SRTSegment(index=2, start_time=2.0, end_time=4.0, text="Second segment")
        video_segment = SRTSegment(index=0, start_time=0.0, end_time=10.0, text="Video text", channel="TestChannel")

        # Previous match from same channel
        prev_video = SRTSegment(index=0, start_time=0.0, end_time=10.0, text="Prev video", channel="TestChannel")
        prev_vo = SRTSegment(index=1, start_time=1.0, end_time=2.0, text="First segment")
        prev_match = Mock(spec=Match)
        prev_match.video_segment = prev_video
        prev_match.voiceover_segment = prev_vo

        conf = MockConfig(source_channel_coherence_boost=0.05)

        adjusted, reason = apply_source_channel_consistency(
            confidence=0.8,
            vo_segment=vo_segment,
            video_segment=video_segment,
            recent_matches=[prev_match],
            current_channel="TestChannel",
            config=conf,
        )

        assert adjusted > 0.8
        assert "source_channel_coherence" in reason
        assert "TestChannel" in reason

    def test_different_channel_no_boost(self):
        """Different channel from previous match should not get a boost."""
        from src.matching.scoring import apply_source_channel_consistency

        vo_segment = SRTSegment(index=2, start_time=2.0, end_time=4.0, text="Second segment")
        video_segment = SRTSegment(index=0, start_time=0.0, end_time=10.0, text="Video text", channel="ChannelB")

        prev_video = SRTSegment(index=0, start_time=0.0, end_time=10.0, text="Prev video", channel="ChannelA")
        prev_vo = SRTSegment(index=1, start_time=1.0, end_time=2.0, text="First segment")
        prev_match = Mock(spec=Match)
        prev_match.video_segment = prev_video
        prev_match.voiceover_segment = prev_vo

        conf = MockConfig(source_channel_coherence_boost=0.05)

        adjusted, reason = apply_source_channel_consistency(
            confidence=0.8,
            vo_segment=vo_segment,
            video_segment=video_segment,
            recent_matches=[prev_match],
            current_channel="ChannelB",
            config=conf,
        )

        assert adjusted == 0.8
        assert reason == ""

    def test_no_recent_matches_no_boost(self):
        """No previous matches should not get a boost."""
        from src.matching.scoring import apply_source_channel_consistency

        vo_segment = SRTSegment(index=2, start_time=2.0, end_time=4.0, text="Second segment")
        video_segment = SRTSegment(index=0, start_time=0.0, end_time=10.0, text="Video text", channel="TestChannel")

        conf = MockConfig(source_channel_coherence_boost=0.05)

        adjusted, reason = apply_source_channel_consistency(
            confidence=0.8,
            vo_segment=vo_segment,
            video_segment=video_segment,
            recent_matches=None,
            current_channel="TestChannel",
            config=conf,
        )

        assert adjusted == 0.8
        assert reason == ""

    def test_no_current_channel_no_boost(self):
        """No current channel should not get a boost."""
        from src.matching.scoring import apply_source_channel_consistency

        vo_segment = SRTSegment(index=2, start_time=2.0, end_time=4.0, text="Second segment")
        video_segment = SRTSegment(index=0, start_time=0.0, end_time=10.0, text="Video text")

        conf = MockConfig(source_channel_coherence_boost=0.05)

        adjusted, reason = apply_source_channel_consistency(
            confidence=0.8,
            vo_segment=vo_segment,
            video_segment=video_segment,
            recent_matches=[],
            current_channel=None,
            config=conf,
        )

        assert adjusted == 0.8
        assert reason == ""

    def test_zero_boost_config_no_boost(self):
        """Zero boost in config should not apply boost."""
        from src.matching.scoring import apply_source_channel_consistency

        vo_segment = SRTSegment(index=2, start_time=2.0, end_time=4.0, text="Second segment")
        video_segment = SRTSegment(index=0, start_time=0.0, end_time=10.0, text="Video text", channel="TestChannel")

        prev_video = SRTSegment(index=0, start_time=0.0, end_time=10.0, text="Prev video", channel="TestChannel")
        prev_vo = SRTSegment(index=1, start_time=1.0, end_time=2.0, text="First segment")
        prev_match = Mock(spec=Match)
        prev_match.video_segment = prev_video
        prev_match.voiceover_segment = prev_vo

        conf = MockConfig(source_channel_coherence_boost=0.0)

        adjusted, reason = apply_source_channel_consistency(
            confidence=0.8,
            vo_segment=vo_segment,
            video_segment=video_segment,
            recent_matches=[prev_match],
            current_channel="TestChannel",
            config=conf,
        )

        assert adjusted == 0.8
        assert reason == ""

    def test_default_boost_value(self):
        """Default boost should be 0.05 when config not provided."""
        from src.matching.scoring import apply_source_channel_consistency, _DEFAULT_SOURCE_CHANNEL_COHERENCE_BOOST

        vo_segment = SRTSegment(index=2, start_time=2.0, end_time=4.0, text="Second segment")
        video_segment = SRTSegment(index=0, start_time=0.0, end_time=10.0, text="Video text", channel="TestChannel")

        prev_video = SRTSegment(index=0, start_time=0.0, end_time=10.0, text="Prev video", channel="TestChannel")
        prev_vo = SRTSegment(index=1, start_time=1.0, end_time=2.0, text="First segment")
        prev_match = Mock(spec=Match)
        prev_match.video_segment = prev_video
        prev_match.voiceover_segment = prev_vo

        adjusted, reason = apply_source_channel_consistency(
            confidence=0.8,
            vo_segment=vo_segment,
            video_segment=video_segment,
            recent_matches=[prev_match],
            current_channel="TestChannel",
            config=None,
        )

        expected = 0.8 + _DEFAULT_SOURCE_CHANNEL_COHERENCE_BOOST
        assert adjusted == expected

    def test_previous_match_no_channel_no_boost(self):
        """Previous match with no channel should not get a boost."""
        from src.matching.scoring import apply_source_channel_consistency

        vo_segment = SRTSegment(index=2, start_time=2.0, end_time=4.0, text="Second segment")
        video_segment = SRTSegment(index=0, start_time=0.0, end_time=10.0, text="Video text", channel="TestChannel")

        # Previous match has no channel
        prev_video = SRTSegment(index=0, start_time=0.0, end_time=10.0, text="Prev video")
        prev_vo = SRTSegment(index=1, start_time=1.0, end_time=2.0, text="First segment")
        prev_match = Mock(spec=Match)
        prev_match.video_segment = prev_video
        prev_match.voiceover_segment = prev_vo

        conf = MockConfig(source_channel_coherence_boost=0.05)

        adjusted, reason = apply_source_channel_consistency(
            confidence=0.8,
            vo_segment=vo_segment,
            video_segment=video_segment,
            recent_matches=[prev_match],
            current_channel="TestChannel",
            config=conf,
        )

        assert adjusted == 0.8
        assert reason == ""

    def test_boost_capped_at_1_0(self):
        """Boost should not exceed 1.0."""
        from src.matching.scoring import apply_source_channel_consistency

        vo_segment = SRTSegment(index=2, start_time=2.0, end_time=4.0, text="Second segment")
        video_segment = SRTSegment(index=0, start_time=0.0, end_time=10.0, text="Video text", channel="TestChannel")

        prev_video = SRTSegment(index=0, start_time=0.0, end_time=10.0, text="Prev video", channel="TestChannel")
        prev_vo = SRTSegment(index=1, start_time=1.0, end_time=2.0, text="First segment")
        prev_match = Mock(spec=Match)
        prev_match.video_segment = prev_video
        prev_match.voiceover_segment = prev_vo

        conf = MockConfig(source_channel_coherence_boost=0.5)

        adjusted, reason = apply_source_channel_consistency(
            confidence=0.9,
            vo_segment=vo_segment,
            video_segment=video_segment,
            recent_matches=[prev_match],
            current_channel="TestChannel",
            config=conf,
        )

        assert adjusted == 1.0

    def test_same_channel_selected_over_different_channel(self):
        """Test that same-channel video is preferred over different-channel video.

        This test verifies that source channel consistency actually improves
        match quality by demonstrating that a same-channel video wins when
        base confidence scores are equal.
        """
        from src.matching.scoring import apply_source_channel_consistency

        vo_segment = SRTSegment(index=2, start_time=2.0, end_time=4.0, text="Test segment")

        # Previous match was from TestChannel
        prev_video = SRTSegment(index=0, start_time=0.0, end_time=10.0, text="Prev video", channel="TestChannel")
        prev_vo = SRTSegment(index=1, start_time=1.0, end_time=2.0, text="First segment")
        prev_match = Mock(spec=Match)
        prev_match.video_segment = prev_video
        prev_match.voiceover_segment = prev_vo

        conf = MockConfig(source_channel_coherence_boost=0.05)

        # Candidate A: same channel as previous match
        video_a = SRTSegment(index=0, start_time=0.0, end_time=10.0, text="Video A", channel="TestChannel")
        adjusted_a, _ = apply_source_channel_consistency(
            confidence=0.8,
            vo_segment=vo_segment,
            video_segment=video_a,
            recent_matches=[prev_match],
            current_channel="TestChannel",
            config=conf,
        )

        # Candidate B: different channel from previous match
        video_b = SRTSegment(index=0, start_time=0.0, end_time=10.0, text="Video B", channel="OtherChannel")
        adjusted_b, _ = apply_source_channel_consistency(
            confidence=0.8,
            vo_segment=vo_segment,
            video_segment=video_b,
            recent_matches=[prev_match],
            current_channel="OtherChannel",
            config=conf,
        )

        # Same-channel video should have higher adjusted confidence
        assert adjusted_a > adjusted_b, (
            f"Same-channel video (adjusted: {adjusted_a}) should have higher confidence "
            f"than different-channel video (adjusted: {adjusted_b})"
        )
        # The boost should be approximately 0.05
        assert abs(adjusted_a - 0.85) < 0.001  # 0.8 + 0.05 (floating point)
        assert adjusted_b == 0.8   # no boost
