"""
Tests for keyword match highlighting in match reasoning (US-008).

Tests extraction of matching keywords between voiceover and video transcript
that are populated in MatchResult.matched_keywords.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure src is in path
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.utils import MatchResult, SRTSegment, Match


# =============================================================================
# Test MatchResult matched_keywords Field
# =============================================================================

class TestMatchResultMatchedKeywords:
    """Tests for MatchResult.matched_keywords field existence and defaults."""

    def test_matched_keywords_field_exists(self):
        """MatchResult should have matched_keywords field."""
        # Create minimal MatchResult
        match = Match(
            voiceover_segment=SRTSegment(index=0, start_time=0, end_time=1, text="test"),
            video_segment=SRTSegment(index=0, start_time=0, end_time=1, text="test"),
            video_scene=None,
            confidence=0.8,
            reasoning="test"
        )
        result = MatchResult(primary_match=match)
        assert hasattr(result, 'matched_keywords')

    def test_matched_keywords_default_empty_list(self):
        """matched_keywords should default to empty list."""
        match = Match(
            voiceover_segment=SRTSegment(index=0, start_time=0, end_time=1, text="test"),
            video_segment=SRTSegment(index=0, start_time=0, end_time=1, text="test"),
            video_scene=None,
            confidence=0.8,
            reasoning="test"
        )
        result = MatchResult(primary_match=match)
        assert result.matched_keywords == []

    def test_matched_keywords_can_be_set(self):
        """matched_keywords should accept list of strings."""
        match = Match(
            voiceover_segment=SRTSegment(index=0, start_time=0, end_time=1, text="test"),
            video_segment=SRTSegment(index=0, start_time=0, end_time=1, text="test"),
            video_scene=None,
            confidence=0.8,
            reasoning="test"
        )
        keywords = ['earthquake', 'disaster', 'rescue']
        result = MatchResult(primary_match=match, matched_keywords=keywords)
        assert result.matched_keywords == keywords

    def test_matched_keywords_is_list_type(self):
        """matched_keywords should be a list."""
        match = Match(
            voiceover_segment=SRTSegment(index=0, start_time=0, end_time=1, text="test"),
            video_segment=SRTSegment(index=0, start_time=0, end_time=1, text="test"),
            video_scene=None,
            confidence=0.8,
            reasoning="test"
        )
        result = MatchResult(primary_match=match, matched_keywords=['test'])
        assert isinstance(result.matched_keywords, list)


# =============================================================================
# Test Extract Matched Keywords Helper
# =============================================================================

class TestExtractMatchedKeywordsHelper:
    """Tests for TieredMatcher._extract_matched_keywords() helper."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config for TieredMatcher."""
        config = MagicMock()
        config.matching = MagicMock()
        config.matching.gemini_model = 'gemini-2.0-flash'
        config.matching.min_confidence = 0.5
        config.matching.embedding_candidates = 10
        config.matching.high_confidence_threshold = 0.85
        config.matching.low_confidence_threshold = 0.3
        config.matching.max_clip_reuse = 2
        config.matching.reuse_penalty = 0.2
        config.matching.primary_provider = 'gemini'
        config.matching.secondary_provider = None
        config.matching.use_local_for_review = False
        config.matching.chapter_matching_enabled = False
        config.matching.topic_mismatch_penalty = 0.15
        config.matching.location_matching = None
        config.gemini_api_key = None
        config.anthropic_api_key = None
        return config

    def test_helper_function_exists(self, mock_config):
        """_extract_matched_keywords should exist on TieredMatcher."""
        from src.matching.tiered_matcher import TieredMatcher
        matcher = TieredMatcher(config=mock_config)
        assert hasattr(matcher, '_extract_matched_keywords')
        assert callable(matcher._extract_matched_keywords)

    def test_helper_returns_list(self, mock_config):
        """_extract_matched_keywords should return a list."""
        from src.matching.tiered_matcher import TieredMatcher
        matcher = TieredMatcher(config=mock_config)

        vo_seg = SRTSegment(index=0, start_time=0, end_time=1, text="test text")
        video_seg = SRTSegment(index=0, start_time=0, end_time=1, text="test text")

        result = matcher._extract_matched_keywords(vo_seg, video_seg)
        assert isinstance(result, list)

    def test_helper_finds_common_words(self, mock_config):
        """Should find words common to both segments."""
        from src.matching.tiered_matcher import TieredMatcher
        matcher = TieredMatcher(config=mock_config)

        vo_seg = SRTSegment(
            index=0, start_time=0, end_time=5,
            text="The earthquake caused massive destruction in the city"
        )
        video_seg = SRTSegment(
            index=0, start_time=0, end_time=5,
            text="Earthquake damage visible in the devastated city center"
        )

        result = matcher._extract_matched_keywords(vo_seg, video_seg)
        assert 'earthquake' in result
        assert 'city' in result

    def test_helper_uses_keywords_list(self, mock_config):
        """Should use keywords list if available."""
        from src.matching.tiered_matcher import TieredMatcher
        matcher = TieredMatcher(config=mock_config)

        vo_seg = SRTSegment(
            index=0, start_time=0, end_time=5,
            text="Something happened",
            keywords=['disaster', 'emergency', 'rescue']
        )
        video_seg = SRTSegment(
            index=0, start_time=0, end_time=5,
            text="Something else",
            keywords=['disaster', 'relief', 'rescue']
        )

        result = matcher._extract_matched_keywords(vo_seg, video_seg)
        assert 'disaster' in result
        assert 'rescue' in result


# =============================================================================
# Test Keyword Matching Logic
# =============================================================================

class TestKeywordMatchingLogic:
    """Tests for keyword matching algorithm behavior."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config for TieredMatcher."""
        config = MagicMock()
        config.matching = MagicMock()
        config.matching.gemini_model = 'gemini-2.0-flash'
        config.matching.min_confidence = 0.5
        config.matching.embedding_candidates = 10
        config.matching.high_confidence_threshold = 0.85
        config.matching.low_confidence_threshold = 0.3
        config.matching.max_clip_reuse = 2
        config.matching.reuse_penalty = 0.2
        config.matching.primary_provider = 'gemini'
        config.matching.secondary_provider = None
        config.matching.use_local_for_review = False
        config.matching.chapter_matching_enabled = False
        config.matching.topic_mismatch_penalty = 0.15
        config.matching.location_matching = None
        config.gemini_api_key = None
        config.anthropic_api_key = None
        return config

    def test_filters_common_words(self, mock_config):
        """Should filter out common stopwords."""
        from src.matching.tiered_matcher import TieredMatcher
        matcher = TieredMatcher(config=mock_config)

        vo_seg = SRTSegment(
            index=0, start_time=0, end_time=5,
            text="The building collapsed from the earthquake"
        )
        video_seg = SRTSegment(
            index=0, start_time=0, end_time=5,
            text="The building was destroyed in the disaster"
        )

        result = matcher._extract_matched_keywords(vo_seg, video_seg)
        # 'the' and 'from' should be filtered
        assert 'the' not in result
        assert 'from' not in result
        # 'building' should be included
        assert 'building' in result

    def test_case_insensitive_matching(self, mock_config):
        """Matching should be case-insensitive."""
        from src.matching.tiered_matcher import TieredMatcher
        matcher = TieredMatcher(config=mock_config)

        vo_seg = SRTSegment(
            index=0, start_time=0, end_time=5,
            text="EARTHQUAKE damage reported"
        )
        video_seg = SRTSegment(
            index=0, start_time=0, end_time=5,
            text="earthquake destruction visible"
        )

        result = matcher._extract_matched_keywords(vo_seg, video_seg)
        assert 'earthquake' in result

    def test_filters_short_words(self, mock_config):
        """Should filter words shorter than minimum length."""
        from src.matching.tiered_matcher import TieredMatcher
        matcher = TieredMatcher(config=mock_config)

        vo_seg = SRTSegment(
            index=0, start_time=0, end_time=5,
            text="A big cat sat on mat"
        )
        video_seg = SRTSegment(
            index=0, start_time=0, end_time=5,
            text="A cat and dog on mat"
        )

        result = matcher._extract_matched_keywords(vo_seg, video_seg)
        # Very short words like 'on', 'a' should be filtered
        assert 'on' not in result or len('on') >= 3

    def test_returns_sorted_list(self, mock_config):
        """Should return keywords sorted alphabetically."""
        from src.matching.tiered_matcher import TieredMatcher
        matcher = TieredMatcher(config=mock_config)

        vo_seg = SRTSegment(
            index=0, start_time=0, end_time=5,
            text="zebra apple banana",
            keywords=['zebra', 'apple', 'banana']
        )
        video_seg = SRTSegment(
            index=0, start_time=0, end_time=5,
            text="zebra apple banana",
            keywords=['zebra', 'apple', 'banana']
        )

        result = matcher._extract_matched_keywords(vo_seg, video_seg)
        assert result == sorted(result)

    def test_no_duplicates_in_result(self, mock_config):
        """Should not have duplicate keywords."""
        from src.matching.tiered_matcher import TieredMatcher
        matcher = TieredMatcher(config=mock_config)

        vo_seg = SRTSegment(
            index=0, start_time=0, end_time=5,
            text="earthquake earthquake earthquake",
            keywords=['earthquake']
        )
        video_seg = SRTSegment(
            index=0, start_time=0, end_time=5,
            text="earthquake aftermath",
            keywords=['earthquake']
        )

        result = matcher._extract_matched_keywords(vo_seg, video_seg)
        assert len(result) == len(set(result))


# =============================================================================
# Test Empty and Edge Cases
# =============================================================================

class TestKeywordEdgeCases:
    """Tests for edge cases in keyword extraction."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config for TieredMatcher."""
        config = MagicMock()
        config.matching = MagicMock()
        config.matching.gemini_model = 'gemini-2.0-flash'
        config.matching.min_confidence = 0.5
        config.matching.embedding_candidates = 10
        config.matching.high_confidence_threshold = 0.85
        config.matching.low_confidence_threshold = 0.3
        config.matching.max_clip_reuse = 2
        config.matching.reuse_penalty = 0.2
        config.matching.primary_provider = 'gemini'
        config.matching.secondary_provider = None
        config.matching.use_local_for_review = False
        config.matching.chapter_matching_enabled = False
        config.matching.topic_mismatch_penalty = 0.15
        config.matching.location_matching = None
        config.gemini_api_key = None
        config.anthropic_api_key = None
        return config

    def test_empty_text_returns_empty_list(self, mock_config):
        """Empty text should return empty list."""
        from src.matching.tiered_matcher import TieredMatcher
        matcher = TieredMatcher(config=mock_config)

        vo_seg = SRTSegment(index=0, start_time=0, end_time=1, text="")
        video_seg = SRTSegment(index=0, start_time=0, end_time=1, text="")

        result = matcher._extract_matched_keywords(vo_seg, video_seg)
        assert result == []

    def test_no_common_words_returns_empty_list(self, mock_config):
        """No common words should return empty list."""
        from src.matching.tiered_matcher import TieredMatcher
        matcher = TieredMatcher(config=mock_config)

        vo_seg = SRTSegment(
            index=0, start_time=0, end_time=5,
            text="earthquake disaster rescue"
        )
        video_seg = SRTSegment(
            index=0, start_time=0, end_time=5,
            text="cooking recipe kitchen"
        )

        result = matcher._extract_matched_keywords(vo_seg, video_seg)
        assert result == []

    def test_handles_punctuation(self, mock_config):
        """Should handle punctuation in text."""
        from src.matching.tiered_matcher import TieredMatcher
        matcher = TieredMatcher(config=mock_config)

        vo_seg = SRTSegment(
            index=0, start_time=0, end_time=5,
            text="The earthquake, devastating and powerful, struck!"
        )
        video_seg = SRTSegment(
            index=0, start_time=0, end_time=5,
            text="Earthquake: the aftermath was devastating."
        )

        result = matcher._extract_matched_keywords(vo_seg, video_seg)
        assert 'earthquake' in result
        assert 'devastating' in result

    def test_handles_none_keywords_list(self, mock_config):
        """Should handle None keywords list gracefully."""
        from src.matching.tiered_matcher import TieredMatcher
        matcher = TieredMatcher(config=mock_config)

        vo_seg = SRTSegment(
            index=0, start_time=0, end_time=5,
            text="earthquake damage"
        )
        vo_seg.keywords = None
        video_seg = SRTSegment(
            index=0, start_time=0, end_time=5,
            text="earthquake destruction"
        )
        video_seg.keywords = None

        # Should not raise exception
        result = matcher._extract_matched_keywords(vo_seg, video_seg)
        assert isinstance(result, list)


# =============================================================================
# Test Checkpoint Serialization
# =============================================================================

class TestCheckpointSerialization:
    """Tests for matched_keywords in checkpoint output."""

    def test_checkpoint_includes_matched_keywords(self):
        """Checkpoint match data should include matched_keywords field."""
        # Simulate what MatchStage does for serialization
        match_result = MagicMock()
        match_result.primary_match = MagicMock()
        match_result.primary_match.video_segment = MagicMock()
        match_result.primary_match.video_segment.source_file = 'test.mp4'
        match_result.primary_match.video_segment.start_time = 0.0
        match_result.primary_match.confidence = 0.85
        match_result.confidence_variance = 0.05
        match_result.matched_keywords = ['earthquake', 'disaster']

        # Simulate serialization logic from MatchStage
        serialized = {
            'segment_index': 0,
            'source_file': match_result.primary_match.video_segment.source_file,
            'start_time': float(match_result.primary_match.video_segment.start_time),
            'confidence': float(match_result.primary_match.confidence),
            'confidence_variance': float(match_result.confidence_variance),
            'matched_keywords': list(match_result.matched_keywords) if match_result.matched_keywords else []
        }

        assert 'matched_keywords' in serialized
        assert serialized['matched_keywords'] == ['earthquake', 'disaster']

    def test_checkpoint_empty_keywords(self):
        """Empty matched_keywords should serialize as empty list."""
        match_result = MagicMock()
        match_result.matched_keywords = []

        serialized_keywords = list(match_result.matched_keywords) if match_result.matched_keywords else []
        assert serialized_keywords == []

    def test_checkpoint_handles_none_keywords(self):
        """None matched_keywords should serialize as empty list."""
        matched_kws = None
        serialized_keywords = list(matched_kws) if matched_kws else []
        assert serialized_keywords == []


# =============================================================================
# Test Real World Scenarios
# =============================================================================

class TestRealWorldScenarios:
    """Tests with realistic voiceover/video content."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config for TieredMatcher."""
        config = MagicMock()
        config.matching = MagicMock()
        config.matching.gemini_model = 'gemini-2.0-flash'
        config.matching.min_confidence = 0.5
        config.matching.embedding_candidates = 10
        config.matching.high_confidence_threshold = 0.85
        config.matching.low_confidence_threshold = 0.3
        config.matching.max_clip_reuse = 2
        config.matching.reuse_penalty = 0.2
        config.matching.primary_provider = 'gemini'
        config.matching.secondary_provider = None
        config.matching.use_local_for_review = False
        config.matching.chapter_matching_enabled = False
        config.matching.topic_mismatch_penalty = 0.15
        config.matching.location_matching = None
        config.gemini_api_key = None
        config.anthropic_api_key = None
        return config

    def test_documentary_narration_match(self, mock_config):
        """Test with documentary-style narration."""
        from src.matching.tiered_matcher import TieredMatcher
        matcher = TieredMatcher(config=mock_config)

        vo_seg = SRTSegment(
            index=0, start_time=0, end_time=10,
            text="The ancient pyramids of Giza stand as monuments to Egypt's glorious past",
            keywords=['pyramids', 'giza', 'egypt', 'ancient', 'monuments']
        )
        video_seg = SRTSegment(
            index=0, start_time=0, end_time=10,
            text="Footage of pyramids at sunrise in Giza Egypt",
            keywords=['pyramids', 'giza', 'egypt', 'sunrise']
        )

        result = matcher._extract_matched_keywords(vo_seg, video_seg)
        assert 'pyramids' in result
        assert 'giza' in result
        assert 'egypt' in result

    def test_nature_documentary_match(self, mock_config):
        """Test with nature documentary content."""
        from src.matching.tiered_matcher import TieredMatcher
        matcher = TieredMatcher(config=mock_config)

        vo_seg = SRTSegment(
            index=0, start_time=0, end_time=8,
            text="The African elephant is the largest land animal on Earth",
            keywords=['elephant', 'african', 'animal', 'earth']
        )
        video_seg = SRTSegment(
            index=0, start_time=0, end_time=8,
            text="Elephant herd walking across African savanna",
            keywords=['elephant', 'herd', 'african', 'savanna']
        )

        result = matcher._extract_matched_keywords(vo_seg, video_seg)
        assert 'elephant' in result
        assert 'african' in result

    def test_cooking_show_match(self, mock_config):
        """Test with cooking show content."""
        from src.matching.tiered_matcher import TieredMatcher
        matcher = TieredMatcher(config=mock_config)

        vo_seg = SRTSegment(
            index=0, start_time=0, end_time=6,
            text="Now we add the garlic and onions to the sizzling pan",
            keywords=['garlic', 'onions', 'pan', 'cooking']
        )
        video_seg = SRTSegment(
            index=0, start_time=0, end_time=6,
            text="Chef adding garlic and onions to hot pan",
            keywords=['chef', 'garlic', 'onions', 'pan']
        )

        result = matcher._extract_matched_keywords(vo_seg, video_seg)
        assert 'garlic' in result
        assert 'onions' in result


# =============================================================================
# Test Integration with MatchResult
# =============================================================================

class TestMatchResultIntegration:
    """Tests for matched_keywords integration with full MatchResult."""

    def test_match_result_with_all_fields(self):
        """MatchResult should work with all fields populated."""
        from src.utils import AlternativeMatch, StrategyMatch

        vo_seg = SRTSegment(index=0, start_time=0, end_time=5, text="test voiceover")
        video_seg = SRTSegment(index=0, start_time=0, end_time=5, text="test video")

        match = Match(
            voiceover_segment=vo_seg,
            video_segment=video_seg,
            video_scene=None,
            confidence=0.85,
            reasoning="Good match"
        )

        result = MatchResult(
            primary_match=match,
            alternatives=[],
            secondary_matches=[],
            strategy_matches=[],
            has_gap=False,
            gap_reason="",
            confidence_variance=0.05,
            matched_keywords=['test']
        )

        assert result.primary_match == match
        assert result.matched_keywords == ['test']
        assert result.confidence_variance == 0.05
        assert result.has_gap is False

    def test_match_result_defaults(self):
        """MatchResult should have sensible defaults."""
        match = Match(
            voiceover_segment=SRTSegment(index=0, start_time=0, end_time=1, text="t"),
            video_segment=SRTSegment(index=0, start_time=0, end_time=1, text="t"),
            video_scene=None,
            confidence=0.8,
            reasoning="test"
        )

        result = MatchResult(primary_match=match)

        assert result.alternatives == []
        assert result.secondary_matches == []
        assert result.strategy_matches == []
        assert result.has_gap is False
        assert result.gap_reason == ""
        assert result.confidence_variance == 0.0
        assert result.matched_keywords == []
