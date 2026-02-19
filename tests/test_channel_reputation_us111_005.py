"""
Unit tests for Channel Reputation and Engagement Context Signals (US-111-005).

Tests the _apply_channel_reputation method in TieredMatcher.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from src.utils import SRTSegment


# Full mock config matching what TieredMatcher expects
@dataclass
class MockMatchingConfig:
    gemini_model: str = "gemini-2.0-flash"
    anthropic_model: str = "claude-3-haiku-20240307"
    ollama_model: str = "llama3.2"
    ollama_host: str = "http://localhost:11434"
    primary_provider: str = "gemini"
    secondary_provider: str = ""
    use_local_for_review: bool = False
    min_confidence: float = 0.3
    embedding_candidates: int = 10
    high_confidence_threshold: float = 0.85
    low_confidence_threshold: float = 0.5
    skip_llm_threshold: float = 0.70
    ambiguous_threshold: float = 0.6
    confidence_threshold: float = 0.3
    max_clip_reuse: int = 10
    reuse_penalty: float = 0.01
    chapter_matching_enabled: bool = False
    topic_mismatch_penalty: float = 0.15
    location_matching: None = None
    cache_llm_responses: bool = False
    face_preference: str = "neutral"
    caption_quality_adjustment_enabled: bool = False
    timing_penalty_enabled: bool = False
    max_consecutive_same_source: int = 3
    consecutive_source_penalty: float = 0.05
    adaptive_threshold_enabled: bool = False
    current_project_boost: float = 0.0
    broll_boost: float = 0.0
    obvious_match_min_confidence: float = 0.0
    # Channel reputation config (US-111-005)
    channel_reputation_enabled: bool = True
    channel_reputation_boost: float = 0.02
    channel_reputation_threshold: int = 100000
    view_count_context_weight: float = 0.01
    view_count_boost_threshold: int = 1000000


@dataclass
class MockOutputConfig:
    num_alternatives: int = 2
    secondary_matches_enabled: bool = False
    strategy_matches_enabled: bool = False


@dataclass
class MockConfig:
    matching: MockMatchingConfig = field(default_factory=MockMatchingConfig)
    output: MockOutputConfig = field(default_factory=MockOutputConfig)
    gemini_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    negative_matching: MagicMock = field(default_factory=lambda: MagicMock(enabled=False))


def create_mock_segment(start: float = 0.0, end: float = 5.0) -> SRTSegment:
    """Create a mock SRTSegment for testing."""
    seg = SRTSegment(
        index=1,
        start_time=start,
        end_time=end,
        text="Test voiceover segment",
    )
    seg.source_file = "test_video_id"
    return seg


class TestChannelReputation:
    """Tests for channel reputation boost functionality."""

    def _create_matcher_with_metadata(self, channel_data: Dict[str, Any]):
        """Create a TieredMatcher instance with mocked video metadata."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()
        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            matcher = TieredMatcher(
                config=config,
                cache=MagicMock(),
                video_metadata={"test_video_id": channel_data},
            )
        return matcher

    def test_no_channel_data_returns_no_boost(self):
        """Test that videos without channel data get no boost."""
        matcher = self._create_matcher_with_metadata({})

        seg = create_mock_segment()
        confidence_breakdown = []

        result, reason = matcher._apply_channel_reputation(
            0.5, seg, confidence_breakdown
        )

        assert result == 0.5
        assert reason == "no_channel_data"

    def test_disabled_config_returns_no_boost(self):
        """Test that disabled config returns no boost."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()
        config.matching.channel_reputation_enabled = False

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            matcher = TieredMatcher(
                config=config,
                cache=MagicMock(),
                video_metadata={"test_video_id": {"subscriber_count": 500000}},
            )

        seg = create_mock_segment()
        confidence_breakdown = []

        result, reason = matcher._apply_channel_reputation(
            0.5, seg, confidence_breakdown
        )

        assert result == 0.5
        assert reason == "disabled"

    def test_high_subscriber_channel_gets_full_boost(self):
        """Test that high-subscriber channels get full boost."""
        matcher = self._create_matcher_with_metadata({
            "subscriber_count": 500000,
            "view_count": 100000
        })

        seg = create_mock_segment()
        confidence_breakdown = []

        result, reason = matcher._apply_channel_reputation(
            0.5, seg, confidence_breakdown
        )

        # Should get full boost (0.02) for 500K subscribers (above 100K threshold)
        assert result == 0.52
        assert "high_subscribers" in reason

    def test_moderate_subscriber_channel_gets_partial_boost(self):
        """Test that moderate-subscriber channels get partial boost."""
        matcher = self._create_matcher_with_metadata({
            "subscriber_count": 50000,  # 50K - between 10K and 100K
            "view_count": 100000
        })

        seg = create_mock_segment()
        confidence_breakdown = []

        result, reason = matcher._apply_channel_reputation(
            0.5, seg, confidence_breakdown
        )

        # Should get 30% of max boost (0.02 * 0.3 = 0.006)
        expected = 0.5 + (0.02 * 0.3)
        assert abs(result - expected) < 0.001
        assert "moderate_subscribers" in reason

    def test_low_subscriber_channel_gets_no_boost(self):
        """Test that low-subscriber channels get no boost."""
        matcher = self._create_matcher_with_metadata({
            "subscriber_count": 5000,  # Below 10K threshold
            "view_count": 10000
        })

        seg = create_mock_segment()
        confidence_breakdown = []

        result, reason = matcher._apply_channel_reputation(
            0.5, seg, confidence_breakdown
        )

        assert result == 0.5
        assert reason == "no_boost_triggers"

    def test_viral_video_gets_view_count_boost(self):
        """Test that viral videos (high view count) get additional boost."""
        matcher = self._create_matcher_with_metadata({
            "subscriber_count": 0,  # Zero = no subscriber boost
            "view_count": 2000000  # Above 1M threshold
        })

        seg = create_mock_segment()
        confidence_breakdown = []

        result, reason = matcher._apply_channel_reputation(
            0.5, seg, confidence_breakdown
        )

        # Should get view count boost (0.01) since above 1M threshold
        assert result == 0.51
        assert "high_views" in reason

    def test_combined_subscriber_and_view_boost(self):
        """Test that both subscriber and view count boosts can apply."""
        matcher = self._create_matcher_with_metadata({
            "subscriber_count": 500000,  # Above 100K - full subscriber boost
            "view_count": 2000000  # Above 1M - view count boost
        })

        seg = create_mock_segment()
        confidence_breakdown = []

        result, reason = matcher._apply_channel_reputation(
            0.5, seg, confidence_breakdown
        )

        # Should get both boosts: 0.02 (subscriber) + 0.01 (views) = 0.03
        assert result == 0.53
        assert "high_subscribers" in reason
        assert "high_views" in reason

    def test_view_count_boost_disabled_when_weight_zero(self):
        """Test that view count boost is disabled when weight is 0."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()
        config.matching.view_count_context_weight = 0.0  # Disable view count

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            matcher = TieredMatcher(
                config=config,
                cache=MagicMock(),
                video_metadata={
                    "test_video_id": {
                        "subscriber_count": 0,  # Zero = no subscriber boost
                        "view_count": 2000000
                    }
                },
            )

        seg = create_mock_segment()
        confidence_breakdown = []

        result, reason = matcher._apply_channel_reputation(
            0.5, seg, confidence_breakdown
        )

        # Should NOT get view count boost (disabled)
        assert result == 0.5
        assert reason == "no_boost_triggers"

    def test_boost_caps_at_max_confidence(self):
        """Test that boost doesn't exceed 1.0 confidence."""
        matcher = self._create_matcher_with_metadata({
            "subscriber_count": 500000,
            "view_count": 2000000
        })

        seg = create_mock_segment()
        confidence_breakdown = []

        # Test with high initial confidence
        result, reason = matcher._apply_channel_reputation(
            0.99, seg, confidence_breakdown
        )

        # Should cap at 1.0
        assert result == 1.0

    def test_graceful_handling_missing_metadata_entry(self):
        """Test graceful handling when video_id not in metadata."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()
        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            matcher = TieredMatcher(
                config=config,
                cache=MagicMock(),
                # Empty metadata
                video_metadata={},
            )

        seg = create_mock_segment()
        confidence_breakdown = []

        result, reason = matcher._apply_channel_reputation(
            0.5, seg, confidence_breakdown
        )

        assert result == 0.5
        assert reason == "no_channel_data"

    def test_uses_channel_subscriber_count_alias(self):
        """Test that channel_subscriber_count field is also recognized."""
        matcher = self._create_matcher_with_metadata({
            "channel_subscriber_count": 500000  # Using the alias field
        })

        seg = create_mock_segment()
        confidence_breakdown = []

        result, reason = matcher._apply_channel_reputation(
            0.5, seg, confidence_breakdown
        )

        # Should work with channel_subscriber_count too
        assert result == 0.52
        assert "high_subscribers" in reason

    def test_zero_subscriber_count_ignored(self):
        """Test that zero subscriber count is treated as no data."""
        matcher = self._create_matcher_with_metadata({
            "subscriber_count": 0,  # Zero should be ignored
            "view_count": 1000
        })

        seg = create_mock_segment()
        confidence_breakdown = []

        result, reason = matcher._apply_channel_reputation(
            0.5, seg, confidence_breakdown
        )

        assert result == 0.5
        assert reason == "no_boost_triggers"

    def test_none_subscriber_count_ignored(self):
        """Test that None subscriber count is handled gracefully."""
        matcher = self._create_matcher_with_metadata({
            "subscriber_count": None,
            "view_count": None
        })

        seg = create_mock_segment()
        confidence_breakdown = []

        result, reason = matcher._apply_channel_reputation(
            0.5, seg, confidence_breakdown
        )

        # Both None = no channel data at all
        assert result == 0.5
        assert reason == "no_channel_data"

    def test_high_engagement_low_subscribers_edge_case(self):
        """Test edge case: channel with very high engagement but low subscribers.

        This tests the scenario where a video goes viral (very high views)
        but the channel has low subscriber count. The view count boost should
        still apply even though subscriber count is below threshold.
        """
        # Channel with only 5K subscribers (below 10K partial threshold)
        # but video has 10M views (very high engagement)
        matcher = self._create_matcher_with_metadata({
            "subscriber_count": 5000,  # Below 10K - no subscriber boost at all
            "view_count": 10000000      # 10M views - well above 1M threshold
        })

        seg = create_mock_segment()
        confidence_breakdown = []

        result, reason = matcher._apply_channel_reputation(
            0.5, seg, confidence_breakdown
        )

        # Should get view count boost (0.01) even with low subscribers
        expected = 0.5 + 0.01  # view_count_context_weight
        assert abs(result - expected) < 0.001
        assert "high_views" in reason
        # Should NOT get subscriber boost
        assert "high_subscribers" not in reason
        assert "moderate_subscribers" not in reason


class TestChannelContextEmbedding:
    """Tests for channel context embedding enrichment (US-134-004)."""

    def test_format_subscriber_count_millions(self):
        """Test formatting subscriber count in millions."""
        from src.stages.caption_stage import CaptionStage
        assert CaptionStage._format_subscriber_count(1500000) == "1.5M"
        assert CaptionStage._format_subscriber_count(1000000) == "1.0M"
        assert CaptionStage._format_subscriber_count(2500000) == "2.5M"

    def test_format_subscriber_count_thousands(self):
        """Test formatting subscriber count in thousands."""
        from src.stages.caption_stage import CaptionStage
        assert CaptionStage._format_subscriber_count(500000) == "500K"
        assert CaptionStage._format_subscriber_count(100000) == "100K"
        assert CaptionStage._format_subscriber_count(1500) == "2K"
        assert CaptionStage._format_subscriber_count(500) == "500"

    def test_format_subscriber_count_small(self):
        """Test formatting small subscriber counts."""
        from src.stages.caption_stage import CaptionStage
        assert CaptionStage._format_subscriber_count(100) == "100"
        assert CaptionStage._format_subscriber_count(50) == "50"
        assert CaptionStage._format_subscriber_count(0) == "0"

    def test_detect_channel_category_gaming(self):
        """Test detecting gaming category from channel name and tags."""
        from src.stages.caption_stage import CaptionStage
        # From channel name
        result = CaptionStage._detect_channel_category(
            "GamingChannel", [], ""
        )
        assert result == "gaming"

        # From tags
        result = CaptionStage._detect_channel_category(
            "RandomChannel", ["minecraft", "gaming"], ""
        )
        assert result == "gaming"

    def test_detect_channel_category_music(self):
        """Test detecting music category."""
        from src.stages.caption_stage import CaptionStage
        result = CaptionStage._detect_channel_category(
            "MusicVEVO", ["new song", "album"], ""
        )
        assert result == "music"

        result = CaptionStage._detect_channel_category(
            "RandomChannel", ["lyrics", "cover"], ""
        )
        assert result == "music"

    def test_detect_channel_category_tech(self):
        """Test detecting tech category."""
        from src.stages.caption_stage import CaptionStage
        result = CaptionStage._detect_channel_category(
            "TechReview", [], ""
        )
        assert result == "tech"

        result = CaptionStage._detect_channel_category(
            "RandomChannel", ["unboxing", "smartphone"], ""
        )
        assert result == "tech"

    def test_detect_channel_category_cooking(self):
        """Test detecting cooking category."""
        from src.stages.caption_stage import CaptionStage
        result = CaptionStage._detect_channel_category(
            "ChefJohn", [], ""
        )
        assert result == "cooking"

        result = CaptionStage._detect_channel_category(
            "RandomChannel", ["recipe", "kitchen"], ""
        )
        assert result == "cooking"

    def test_detect_channel_category_no_match(self):
        """Test that no category is detected when there's no match."""
        from src.stages.caption_stage import CaptionStage
        result = CaptionStage._detect_channel_category(
            "RandomChannel", ["random", "stuff"], ""
        )
        assert result is None

    def test_detect_channel_category_from_title(self):
        """Test detecting category from video title."""
        from src.stages.caption_stage import CaptionStage
        result = CaptionStage._detect_channel_category(
            "RandomChannel", [], "How to play guitar tutorial"
        )
        # "play" matches gaming first
        assert result == "gaming"

        result = CaptionStage._detect_channel_category(
            "RandomChannel", [], "NBA Finals highlights"
        )
        assert result == "sports"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--no-cov"])
