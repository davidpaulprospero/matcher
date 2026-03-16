"""
Tests for US-75-002: Wire title_relevance adjustment into TieredMatcher scoring pipeline.

Verifies:
- Standalone apply_title_relevance_adjustment function exists and works
- Graduated boost: 1 match -> +0.03, 2 matches -> +0.05, 3+ matches -> +0.08
- title_relevance appears in confidence_breakdown when video_title has keyword overlap
- title_relevance adjustment is zero (no-op) when video_title is empty or None
- TieredMatcher with video_metadata containing titles produces title_relevance in breakdown
- TieredMatcher without video_metadata produces no title_relevance entry
"""

import pytest
from unittest.mock import patch, MagicMock, Mock
from dataclasses import dataclass, field
from typing import Optional, List

from src.matching.scoring import apply_title_relevance_adjustment
from src.utils import SRTSegment


# ============================================================================
# Test standalone apply_title_relevance_adjustment function
# ============================================================================

class TestApplyTitleRelevanceAdjustment:
    """Test the standalone apply_title_relevance_adjustment function."""

    @pytest.mark.fast
    def test_one_keyword_match_boost_003(self):
        """1 keyword match -> +0.03 boost."""
        vo = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="The history of Tokyo architecture",
            source_file="vo.srt",
        )
        adjusted, reason = apply_title_relevance_adjustment(
            0.70, vo, video_title="Tokyo travel guide for visitors"
        )
        assert adjusted == pytest.approx(0.73, abs=0.001)
        assert "+0.03" in reason
        assert "1 keyword" in reason

    @pytest.mark.fast
    def test_two_keyword_matches_boost_005(self):
        """2 keyword matches -> +0.05 boost."""
        vo = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="The culture and history of Tokyo",
            source_file="vo.srt",
        )
        adjusted, reason = apply_title_relevance_adjustment(
            0.70, vo, video_title="Tokyo culture and modern lifestyle"
        )
        assert adjusted == pytest.approx(0.75, abs=0.001)
        assert "+0.05" in reason
        assert "2 keywords" in reason

    @pytest.mark.fast
    def test_three_plus_keyword_matches_boost_008(self):
        """3+ keyword matches -> +0.08 boost."""
        vo = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="Tokyo culture food traditional Japanese cuisine",
            source_file="vo.srt",
        )
        adjusted, reason = apply_title_relevance_adjustment(
            0.70, vo, video_title="Tokyo culture food documentary highlights"
        )
        assert adjusted == pytest.approx(0.78, abs=0.001)
        assert "+0.08" in reason
        assert "3 keywords" in reason

    @pytest.mark.fast
    def test_no_title_no_boost(self):
        """No boost when video_title is None."""
        vo = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="Tokyo culture food",
            source_file="vo.srt",
        )
        adjusted, reason = apply_title_relevance_adjustment(
            0.70, vo, video_title=None
        )
        assert adjusted == pytest.approx(0.70, abs=0.001)
        assert reason == ""

    @pytest.mark.fast
    def test_empty_title_no_boost(self):
        """No boost when video_title is empty string."""
        vo = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="Tokyo culture food",
            source_file="vo.srt",
        )
        adjusted, reason = apply_title_relevance_adjustment(
            0.70, vo, video_title=""
        )
        assert adjusted == pytest.approx(0.70, abs=0.001)
        assert reason == ""

    @pytest.mark.fast
    def test_no_overlap_no_boost(self):
        """No boost when there's no keyword overlap."""
        vo = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="The history of architecture",
            source_file="vo.srt",
        )
        adjusted, reason = apply_title_relevance_adjustment(
            0.70, vo, video_title="Completely different topic about science"
        )
        assert adjusted == pytest.approx(0.70, abs=0.001)
        assert reason == ""

    @pytest.mark.fast
    def test_confidence_capped_at_1(self):
        """Boost should not exceed 1.0."""
        vo = SRTSegment(
            index=1, start_time=0.0, end_time=10.0,
            text="Tokyo culture food traditional",
            source_file="vo.srt",
        )
        adjusted, reason = apply_title_relevance_adjustment(
            0.98, vo, video_title="Tokyo culture food documentary"
        )
        assert adjusted <= 1.0


# ============================================================================
# Test title_relevance wiring in TieredMatcher
# ============================================================================

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
    skip_llm_threshold: float = 0.9
    ambiguous_threshold: float = 0.6
    confidence_threshold: float = 0.3
    max_clip_reuse: int = 3
    reuse_penalty: float = 0.1
    chapter_matching_enabled: bool = False
    topic_mismatch_penalty: float = 0.15
    location_matching: None = None
    cache_llm_responses: bool = False
    face_preference: str = "neutral"


@dataclass
class MockConfig:
    matching: MockMatchingConfig = field(default_factory=MockMatchingConfig)
    gemini_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    negative_matching: MagicMock = field(default_factory=lambda: MagicMock(enabled=False))
    output: MagicMock = field(default_factory=lambda: MagicMock(num_alternatives=2))


@dataclass
class MockSRTSegment:
    """Mock SRTSegment for testing."""
    source_file: str = "/path/to/video.mp4"
    start_time: float = 0.0
    end_time: float = 10.0
    text: str = "Test segment text"
    keywords: List[str] = field(default_factory=list)
    source: Optional[str] = None
    face_score: Optional[float] = None
    is_broll: bool = False


class TestTieredMatcherTitleRelevanceInBreakdown:
    """Test that TieredMatcher wires title_relevance into confidence_breakdown."""

    @pytest.mark.fast
    def test_title_relevance_in_breakdown_with_video_metadata(self):
        """TieredMatcher with video_metadata containing titles produces title_relevance in breakdown."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()
        config.matching.skip_llm_threshold = 0.8

        video_metadata = {
            "video123": {"title": "Tokyo culture food documentary highlights", "description": ""},
        }

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            matcher = TieredMatcher(config=config, video_metadata=video_metadata)

        # Voiceover with keywords that overlap with the title
        vo_segment = MockSRTSegment(text="Tokyo culture food traditional Japanese cuisine")
        # Candidate segment from video123
        seg = MockSRTSegment(source_file="video123")
        candidates = [(seg, 0.95)]  # High similarity -> skip LLM path

        with patch.object(matcher, 'reuse_tracker') as mock_tracker:
            mock_tracker.can_use.return_value = True
            mock_tracker.adjust_confidence.return_value = 0.95
            mock_tracker.get_usage_count.return_value = 0
            mock_tracker.record_usage.return_value = None

            with patch('src.matching.tiered_matcher.apply_topic_penalty', return_value=(0.95, "")):
                with patch('src.matching.tiered_matcher.apply_broll_boost', return_value=(0.95, "")):
                    with patch('src.matching.tiered_matcher.apply_current_project_boost', return_value=(0.95, "")):
                        result = matcher.match_segment(vo_segment, candidates)

        # Verify title_relevance appears in the breakdown
        breakdown = result.primary_match.confidence_breakdown
        title_entries = [b for b in breakdown if b['component'] == 'title_relevance']
        assert len(title_entries) == 1, (
            f"Expected title_relevance in breakdown, got components: "
            f"{[b['component'] for b in breakdown]}"
        )
        entry = title_entries[0]
        assert entry['adjustment'] > 0  # Positive boost (Tokyo, culture, food overlap)
        assert 'reason' in entry
        assert 'title relevance boost' in entry['reason']

    @pytest.mark.fast
    def test_no_title_relevance_without_video_metadata(self):
        """TieredMatcher without video_metadata produces no title_relevance entry."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()
        config.matching.skip_llm_threshold = 0.8

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            matcher = TieredMatcher(config=config)  # No video_metadata

        vo_segment = MockSRTSegment(text="Tokyo culture food traditional")
        seg = MockSRTSegment(source_file="video123")
        candidates = [(seg, 0.95)]

        with patch.object(matcher, 'reuse_tracker') as mock_tracker:
            mock_tracker.can_use.return_value = True
            mock_tracker.adjust_confidence.return_value = 0.95
            mock_tracker.get_usage_count.return_value = 0
            mock_tracker.record_usage.return_value = None

            with patch('src.matching.tiered_matcher.apply_topic_penalty', return_value=(0.95, "")):
                with patch('src.matching.tiered_matcher.apply_broll_boost', return_value=(0.95, "")):
                    with patch('src.matching.tiered_matcher.apply_current_project_boost', return_value=(0.95, "")):
                        result = matcher.match_segment(vo_segment, candidates)

        # Verify no title_relevance in breakdown (empty reason -> _record_breakdown skips it)
        breakdown = result.primary_match.confidence_breakdown
        title_entries = [b for b in breakdown if b['component'] == 'title_relevance']
        assert len(title_entries) == 0, (
            f"Expected no title_relevance entry without video_metadata, "
            f"but got: {title_entries}"
        )

    @pytest.mark.fast
    def test_title_relevance_in_reasoning_string(self):
        """Title relevance reason appears in the match reasoning string."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()
        config.matching.skip_llm_threshold = 0.8

        video_metadata = {
            "video123": {"title": "Tokyo culture food documentary highlights", "description": ""},
        }

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            matcher = TieredMatcher(config=config, video_metadata=video_metadata)

        vo_segment = MockSRTSegment(text="Tokyo culture food traditional Japanese cuisine")
        seg = MockSRTSegment(source_file="video123")
        candidates = [(seg, 0.95)]

        with patch.object(matcher, 'reuse_tracker') as mock_tracker:
            mock_tracker.can_use.return_value = True
            mock_tracker.adjust_confidence.return_value = 0.95
            mock_tracker.get_usage_count.return_value = 0
            mock_tracker.record_usage.return_value = None

            with patch('src.matching.tiered_matcher.apply_topic_penalty', return_value=(0.95, "")):
                with patch('src.matching.tiered_matcher.apply_broll_boost', return_value=(0.95, "")):
                    with patch('src.matching.tiered_matcher.apply_current_project_boost', return_value=(0.95, "")):
                        result = matcher.match_segment(vo_segment, candidates)

        assert "title relevance boost" in result.primary_match.reasoning

    @pytest.mark.fast
    def test_get_video_title_helper(self):
        """_get_video_title resolves title from video_metadata."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()
        video_metadata = {
            "vid1": {"title": "Test Title", "description": "desc"},
            "vid2": {"title": "Other Title", "description": ""},
        }

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            matcher = TieredMatcher(config=config, video_metadata=video_metadata)

        seg1 = MockSRTSegment(source_file="vid1")
        seg2 = MockSRTSegment(source_file="vid2")
        seg3 = MockSRTSegment(source_file="vid_unknown")

        assert matcher._get_video_title(seg1) == "Test Title"
        assert matcher._get_video_title(seg2) == "Other Title"
        assert matcher._get_video_title(seg3) is None

    @pytest.mark.fast
    def test_get_video_title_empty_metadata(self):
        """_get_video_title returns None when no video_metadata."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            matcher = TieredMatcher(config=config)

        seg = MockSRTSegment(source_file="vid1")
        assert matcher._get_video_title(seg) is None
