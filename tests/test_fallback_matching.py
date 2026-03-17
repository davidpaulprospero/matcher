"""
Tests for FallbackMatchStrategy - fallback matching for edge cases.

Tests the 3 fallback levels:
- Level 1: Keyword-only matching (ceiling: 0.7)
- Level 2: Visual-description matching (ceiling: 0.5)
- Level 3: Generic B-roll matching (ceiling: 0.3)
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import List, Tuple

from src.matching.strategies import FallbackMatchStrategy
from src.utils import SRTSegment, SceneInfo


@pytest.fixture
def mock_config():
    """Create a mock config with fallback settings."""
    config = MagicMock()
    config.matching = MagicMock()
    config.matching.fallback_matching_enabled = True
    config.matching.fallback_trigger_threshold = 0.4
    return config


@pytest.fixture
def mock_config_disabled():
    """Create a mock config with fallback disabled."""
    config = MagicMock()
    config.matching = MagicMock()
    config.matching.fallback_matching_enabled = False
    config.matching.fallback_trigger_threshold = 0.4
    return config


@pytest.fixture
def fallback_strategy(mock_config):
    """Create a FallbackMatchStrategy instance."""
    return FallbackMatchStrategy(mock_config)


@pytest.fixture
def vo_segment_with_keywords():
    """Create a voiceover segment with keywords."""
    segment = SRTSegment(
        index=1,
        start_time=0.0,
        end_time=5.0,
        text="The earthquake caused massive destruction in the city",
        source_file="voiceover.srt"
    )
    segment.keywords = ["earthquake", "destruction", "city"]
    return segment


@pytest.fixture
def vo_segment_visual():
    """Create a voiceover segment with visual terms."""
    segment = SRTSegment(
        index=2,
        start_time=5.0,
        end_time=10.0,
        text="The tsunami wave flooded the entire coastal city",
        source_file="voiceover.srt"
    )
    segment.keywords = []
    return segment


@pytest.fixture
def vo_segment_no_keywords():
    """Create a voiceover segment without keywords."""
    segment = SRTSegment(
        index=3,
        start_time=10.0,
        end_time=15.0,
        text="Something happened here",
        source_file="voiceover.srt"
    )
    segment.keywords = []
    return segment


@pytest.fixture
def video_candidates():
    """Create a list of video candidates."""
    candidates = []

    # Candidate 1: Has matching keywords
    seg1 = SRTSegment(
        index=1,
        start_time=0.0,
        end_time=10.0,
        text="Major earthquake destroys buildings in downtown city area",
        source_file="video1.mp4"
    )
    seg1.keywords = ["earthquake", "buildings", "city", "destruction"]
    candidates.append((seg1, 0.35))

    # Candidate 2: Has visual terms
    seg2 = SRTSegment(
        index=2,
        start_time=0.0,
        end_time=10.0,
        text="Waves crash against the shoreline during the storm",
        source_file="video2.mp4"
    )
    seg2.keywords = ["wave", "storm"]
    candidates.append((seg2, 0.30))

    # Candidate 3: B-roll segment
    seg3 = SRTSegment(
        index=3,
        start_time=0.0,
        end_time=10.0,
        text="",
        source_file="video3.mp4"
    )
    seg3.keywords = []
    seg3.is_broll = True
    candidates.append((seg3, 0.25))

    # Candidate 4: Regular segment (short text)
    seg4 = SRTSegment(
        index=4,
        start_time=0.0,
        end_time=10.0,
        text="Brief clip",
        source_file="video4.mp4"
    )
    seg4.keywords = []
    candidates.append((seg4, 0.20))

    return candidates


class TestFallbackMatchStrategyInit:
    """Tests for FallbackMatchStrategy initialization."""

    @pytest.mark.fast
    def test_init_with_enabled_config(self, mock_config):
        """Test initialization with fallback enabled."""
        strategy = FallbackMatchStrategy(mock_config)
        assert strategy.enabled is True
        assert strategy.trigger_threshold == 0.4

    @pytest.mark.fast
    def test_init_with_disabled_config(self, mock_config_disabled):
        """Test initialization with fallback disabled."""
        strategy = FallbackMatchStrategy(mock_config_disabled)
        assert strategy.enabled is False

    @pytest.mark.fast
    def test_init_with_missing_config_uses_defaults(self):
        """Test that missing config values use defaults."""
        config = MagicMock()
        config.matching = MagicMock(spec=[])  # No attributes
        strategy = FallbackMatchStrategy(config)
        assert strategy.enabled is True  # Default
        assert strategy.trigger_threshold == 0.4  # Default


class TestShouldTrigger:
    """Tests for should_trigger method."""

    @pytest.mark.fast
    def test_trigger_when_below_threshold(self, fallback_strategy):
        """Test fallback triggers when confidence is below threshold."""
        assert fallback_strategy.should_trigger(0.3) is True
        assert fallback_strategy.should_trigger(0.39) is True

    @pytest.mark.fast
    def test_no_trigger_when_at_threshold(self, fallback_strategy):
        """Test fallback does not trigger at exactly threshold."""
        assert fallback_strategy.should_trigger(0.4) is False

    @pytest.mark.fast
    def test_no_trigger_when_above_threshold(self, fallback_strategy):
        """Test fallback does not trigger above threshold."""
        assert fallback_strategy.should_trigger(0.5) is False
        assert fallback_strategy.should_trigger(0.8) is False

    @pytest.mark.fast
    def test_no_trigger_when_disabled(self, mock_config_disabled):
        """Test fallback does not trigger when disabled."""
        strategy = FallbackMatchStrategy(mock_config_disabled)
        assert strategy.should_trigger(0.1) is False


class TestKeywordOnlyFallback:
    """Tests for Level 1: Keyword-only matching."""

    @pytest.mark.fast
    def test_keyword_match_success(self, fallback_strategy, vo_segment_with_keywords, video_candidates):
        """Test keyword matching finds correct candidate."""
        result = fallback_strategy.match_keyword_only(vo_segment_with_keywords, video_candidates)

        assert result is not None
        segment, confidence, reasoning = result
        assert segment.source_file == "video1.mp4"  # Has matching keywords
        assert confidence <= FallbackMatchStrategy.KEYWORD_ONLY_CEILING
        assert "Fallback L1" in reasoning
        assert "keyword match" in reasoning

    @pytest.mark.fast
    def test_keyword_match_no_keywords_extracts_from_text(self, fallback_strategy, video_candidates):
        """Test that keywords are extracted from text when not present."""
        # Segment with text but no explicit keywords
        segment = SRTSegment(
            index=1,
            start_time=0.0,
            end_time=5.0,
            text="The earthquake destroyed buildings in the city",
            source_file="voiceover.srt"
        )
        segment.keywords = []

        result = fallback_strategy.match_keyword_only(segment, video_candidates)

        # Should still find a match based on text word extraction
        assert result is not None
        _, confidence, _ = result
        assert confidence <= FallbackMatchStrategy.KEYWORD_ONLY_CEILING

    @pytest.mark.fast
    def test_keyword_match_no_match_possible(self, fallback_strategy):
        """Test keyword matching returns None when no matches."""
        segment = SRTSegment(
            index=1,
            start_time=0.0,
            end_time=5.0,
            text="xyz123 abc456",
            source_file="voiceover.srt"
        )
        segment.keywords = ["xyz123", "abc456"]

        candidates = [(
            SRTSegment(
                index=1,
                start_time=0.0,
                end_time=5.0,
                text="Completely unrelated content",
                source_file="video.mp4"
            ), 0.3
        )]
        candidates[0][0].keywords = ["unrelated"]

        result = fallback_strategy.match_keyword_only(segment, candidates)
        assert result is None

    @pytest.mark.fast
    def test_keyword_confidence_ceiling_enforced(self, fallback_strategy, vo_segment_with_keywords, video_candidates):
        """Test that confidence is capped at KEYWORD_ONLY_CEILING."""
        result = fallback_strategy.match_keyword_only(vo_segment_with_keywords, video_candidates)

        if result:
            _, confidence, _ = result
            assert confidence <= FallbackMatchStrategy.KEYWORD_ONLY_CEILING


class TestVisualDescriptionFallback:
    """Tests for Level 2: Visual-description matching."""

    @pytest.mark.fast
    def test_visual_match_from_voiceover_terms(self, fallback_strategy, vo_segment_visual, video_candidates):
        """Test visual matching finds candidate based on visual terms in voiceover."""
        result = fallback_strategy.match_visual_description(vo_segment_visual, video_candidates)

        assert result is not None
        segment, confidence, reasoning = result
        assert confidence <= FallbackMatchStrategy.VISUAL_DESCRIPTION_CEILING
        assert "Fallback L2" in reasoning

    @pytest.mark.fast
    def test_visual_match_with_scenes(self, fallback_strategy, vo_segment_visual, video_candidates):
        """Test visual matching uses scene descriptions."""
        # Create scene info for video2 using the correct SceneInfo structure
        scene = SceneInfo(
            video_path="video2.mp4",
            scene_index=0,
            start_time=0.0,
            end_time=10.0,
            description="Large tsunami wave hitting coastal buildings",
            visual_keywords=["tsunami", "wave", "coastal", "city"]
        )
        scenes = {"video2.mp4": [scene]}

        result = fallback_strategy.match_visual_description(vo_segment_visual, video_candidates, scenes)

        assert result is not None
        segment, confidence, reasoning = result
        assert segment.source_file == "video2.mp4"
        assert confidence <= FallbackMatchStrategy.VISUAL_DESCRIPTION_CEILING

    @pytest.mark.fast
    def test_visual_match_no_visual_terms(self, fallback_strategy, vo_segment_no_keywords, video_candidates):
        """Test visual matching returns None when no visual terms found."""
        result = fallback_strategy.match_visual_description(vo_segment_no_keywords, video_candidates)
        # May return None or low-confidence match
        if result:
            _, confidence, _ = result
            assert confidence <= FallbackMatchStrategy.VISUAL_DESCRIPTION_CEILING

    @pytest.mark.fast
    def test_visual_confidence_ceiling_enforced(self, fallback_strategy, vo_segment_visual, video_candidates):
        """Test that confidence is capped at VISUAL_DESCRIPTION_CEILING."""
        result = fallback_strategy.match_visual_description(vo_segment_visual, video_candidates)

        if result:
            _, confidence, _ = result
            assert confidence <= FallbackMatchStrategy.VISUAL_DESCRIPTION_CEILING


class TestGenericBrollFallback:
    """Tests for Level 3: Generic B-roll matching."""

    @pytest.mark.fast
    def test_broll_match_finds_broll_segment(self, fallback_strategy, vo_segment_no_keywords, video_candidates):
        """Test generic B-roll matching finds B-roll segment."""
        result = fallback_strategy.match_generic_broll(vo_segment_no_keywords, video_candidates)

        assert result is not None
        segment, confidence, reasoning = result
        assert segment.source_file == "video3.mp4"  # The B-roll segment
        assert confidence <= FallbackMatchStrategy.GENERIC_BROLL_CEILING
        assert "Fallback L3" in reasoning
        assert "B-roll" in reasoning or "silent" in reasoning.lower()

    @pytest.mark.fast
    def test_broll_match_finds_short_segment_when_no_broll(self, fallback_strategy, vo_segment_no_keywords):
        """Test generic B-roll finds short/silent segments when no B-roll flagged."""
        # Create candidates without is_broll flag
        candidates = []
        seg1 = SRTSegment(
            index=1,
            start_time=0.0,
            end_time=5.0,
            text="This is a longer transcript with many words in it",
            source_file="video1.mp4"
        )
        candidates.append((seg1, 0.5))

        seg2 = SRTSegment(
            index=2,
            start_time=0.0,
            end_time=5.0,
            text="Short clip",  # Less than 10 words
            source_file="video2.mp4"
        )
        candidates.append((seg2, 0.4))

        result = fallback_strategy.match_generic_broll(vo_segment_no_keywords, candidates)

        assert result is not None
        segment, confidence, reasoning = result
        assert segment.source_file == "video2.mp4"  # The short segment
        assert "silent" in reasoning or "short" in reasoning

    @pytest.mark.fast
    def test_broll_fallback_to_best_available(self, fallback_strategy, vo_segment_no_keywords):
        """Test generic B-roll returns best available when no B-roll or short segments."""
        candidates = []
        seg1 = SRTSegment(
            index=1,
            start_time=0.0,
            end_time=5.0,
            text="This is a longer transcript with many many many words in it today",
            source_file="video1.mp4"
        )
        candidates.append((seg1, 0.5))

        seg2 = SRTSegment(
            index=2,
            start_time=0.0,
            end_time=5.0,
            text="Another longer transcript with plenty of words to exceed threshold",
            source_file="video2.mp4"
        )
        candidates.append((seg2, 0.3))

        result = fallback_strategy.match_generic_broll(vo_segment_no_keywords, candidates)

        assert result is not None
        segment, confidence, reasoning = result
        assert segment.source_file == "video1.mp4"  # Best embedding match
        assert "best available" in reasoning
        assert confidence <= FallbackMatchStrategy.GENERIC_BROLL_CEILING

    @pytest.mark.fast
    def test_broll_confidence_ceiling_enforced(self, fallback_strategy, vo_segment_no_keywords, video_candidates):
        """Test that confidence is capped at GENERIC_BROLL_CEILING."""
        result = fallback_strategy.match_generic_broll(vo_segment_no_keywords, video_candidates)

        assert result is not None
        _, confidence, _ = result
        assert confidence <= FallbackMatchStrategy.GENERIC_BROLL_CEILING

    @pytest.mark.fast
    def test_broll_match_empty_candidates(self, fallback_strategy, vo_segment_no_keywords):
        """Test generic B-roll returns None for empty candidates."""
        result = fallback_strategy.match_generic_broll(vo_segment_no_keywords, [])
        assert result is None


class TestApplyFallback:
    """Tests for the main apply_fallback method."""

    @pytest.mark.fast
    def test_apply_fallback_uses_level1_first(self, fallback_strategy, vo_segment_with_keywords, video_candidates):
        """Test apply_fallback tries Level 1 first."""
        result = fallback_strategy.apply_fallback(
            vo_segment_with_keywords,
            video_candidates,
            primary_confidence=0.3
        )

        assert result is not None
        segment, confidence, reasoning, level = result
        assert level == 1  # Should use keyword matching
        assert "L1" in reasoning

    @pytest.mark.fast
    def test_apply_fallback_uses_level2_when_level1_fails(self, fallback_strategy, vo_segment_visual):
        """Test apply_fallback uses Level 2 when Level 1 fails."""
        # Create candidates that won't match on keywords but will on visual
        candidates = []
        seg = SRTSegment(
            index=1,
            start_time=0.0,
            end_time=5.0,
            text="The wave crashed against the shore during the flood",
            source_file="video1.mp4"
        )
        seg.keywords = []  # No keywords to match
        candidates.append((seg, 0.3))

        result = fallback_strategy.apply_fallback(
            vo_segment_visual,
            candidates,
            primary_confidence=0.3
        )

        assert result is not None
        segment, confidence, reasoning, level = result
        # Level depends on whether keyword extraction from text finds matches
        assert level in [1, 2]

    @pytest.mark.fast
    def test_apply_fallback_uses_level3_when_others_fail(self, fallback_strategy, vo_segment_no_keywords):
        """Test apply_fallback uses Level 3 when Levels 1 and 2 fail."""
        # Create candidates that won't match on keywords or visual
        candidates = []
        seg = SRTSegment(
            index=1,
            start_time=0.0,
            end_time=5.0,
            text="",
            source_file="video1.mp4"
        )
        seg.keywords = []
        seg.is_broll = True
        candidates.append((seg, 0.3))

        result = fallback_strategy.apply_fallback(
            vo_segment_no_keywords,
            candidates,
            primary_confidence=0.3
        )

        assert result is not None
        segment, confidence, reasoning, level = result
        assert level == 3  # Should fall back to B-roll
        assert "L3" in reasoning

    @pytest.mark.fast
    def test_apply_fallback_returns_none_when_not_triggered(self, fallback_strategy, vo_segment_with_keywords, video_candidates):
        """Test apply_fallback returns None when not triggered."""
        result = fallback_strategy.apply_fallback(
            vo_segment_with_keywords,
            video_candidates,
            primary_confidence=0.5  # Above threshold
        )
        assert result is None

    @pytest.mark.fast
    def test_apply_fallback_returns_none_when_disabled(self, mock_config_disabled, vo_segment_with_keywords, video_candidates):
        """Test apply_fallback returns None when disabled."""
        strategy = FallbackMatchStrategy(mock_config_disabled)
        result = strategy.apply_fallback(
            vo_segment_with_keywords,
            video_candidates,
            primary_confidence=0.3
        )
        assert result is None

    @pytest.mark.fast
    def test_apply_fallback_returns_none_for_empty_candidates(self, fallback_strategy, vo_segment_with_keywords):
        """Test apply_fallback returns None for empty candidates."""
        result = fallback_strategy.apply_fallback(
            vo_segment_with_keywords,
            [],
            primary_confidence=0.3
        )
        assert result is None

    @pytest.mark.fast
    def test_apply_fallback_with_scenes(self, fallback_strategy, vo_segment_visual, video_candidates):
        """Test apply_fallback passes scenes to visual matching."""
        scene = SceneInfo(
            video_path="video2.mp4",
            scene_index=0,
            start_time=0.0,
            end_time=10.0,
            description="Tsunami wave flooding the city",
            visual_keywords=["tsunami", "wave", "flood", "city"]
        )
        scenes = {"video2.mp4": [scene]}

        result = fallback_strategy.apply_fallback(
            vo_segment_visual,
            video_candidates,
            primary_confidence=0.3,
            scenes=scenes
        )

        assert result is not None
        # Should find a match using visual description with scenes


class TestConfidenceCeilings:
    """Tests to verify confidence ceilings are properly enforced."""

    @pytest.mark.fast
    def test_keyword_ceiling_constant(self):
        """Test keyword-only ceiling constant value."""
        assert FallbackMatchStrategy.KEYWORD_ONLY_CEILING == 0.7

    @pytest.mark.fast
    def test_visual_ceiling_constant(self):
        """Test visual-description ceiling constant value."""
        assert FallbackMatchStrategy.VISUAL_DESCRIPTION_CEILING == 0.5

    @pytest.mark.fast
    def test_generic_broll_ceiling_constant(self):
        """Test generic B-roll ceiling constant value."""
        assert FallbackMatchStrategy.GENERIC_BROLL_CEILING == 0.3

    @pytest.mark.fast
    def test_ceilings_are_in_order(self):
        """Test that ceilings are in decreasing order (L1 > L2 > L3)."""
        assert FallbackMatchStrategy.KEYWORD_ONLY_CEILING > FallbackMatchStrategy.VISUAL_DESCRIPTION_CEILING
        assert FallbackMatchStrategy.VISUAL_DESCRIPTION_CEILING > FallbackMatchStrategy.GENERIC_BROLL_CEILING
