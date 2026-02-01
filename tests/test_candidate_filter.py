"""
Tests for CandidateFilter - US-33-006.

Tests the extracted candidate filtering logic:
- Face preference filtering
- Location-based filtering
- Smart reuse filtering
- Combined filter application
"""

import pytest
from unittest.mock import MagicMock, patch

from src.matching.candidate_filter import (
    CandidateFilter,
    CandidateFilterConfig,
    FilterResult,
)
from src.utils import SRTSegment, ReuseTracker


class MockConfig:
    """Mock config for testing."""

    def __init__(self, face_preference='neutral', location_enabled=False):
        self.matching = MagicMock()
        self.matching.face_preference = face_preference
        self.matching.location_matching = {'enabled': location_enabled}
        self.matching.max_clip_reuse = 3
        self.matching.reuse_penalty = 0.1


def create_segment(
    text="test segment",
    source_file="video1.mp4",
    start_time=0.0,
    end_time=5.0,
    source=None,
    face_score=None
):
    """Create a test segment."""
    seg = SRTSegment(
        index=1,
        start_time=start_time,
        end_time=end_time,
        text=text,
        source_file=source_file
    )
    if source:
        seg.source = source
    if face_score is not None:
        seg.face_score = face_score
    return seg


class TestCandidateFilterInit:
    """Test CandidateFilter initialization."""

    def test_init_with_config(self):
        """Test initialization with explicit config."""
        config = MockConfig(face_preference='more')
        filter = CandidateFilter(config=config)

        assert filter.config == config
        assert filter.face_preference == 'more'
        assert filter.location_matcher is None

    def test_init_without_config(self):
        """Test initialization uses get_config() when no config provided."""
        with patch('src.config.get_config') as mock_get:
            mock_config = MockConfig()
            mock_get.return_value = mock_config

            filter = CandidateFilter()

            assert filter.config == mock_config
            mock_get.assert_called_once()

    def test_init_with_location_matcher(self):
        """Test initialization with LocationMatcher."""
        config = MockConfig()
        mock_location_matcher = MagicMock()

        filter = CandidateFilter(config=config, location_matcher=mock_location_matcher)

        assert filter.location_matcher == mock_location_matcher


class TestFacePreferenceFilter:
    """Test face preference filtering."""

    def test_neutral_preference_no_filter(self):
        """Neutral preference should return candidates unchanged."""
        config = MockConfig(face_preference='neutral')
        filter = CandidateFilter(config=config)

        candidates = [
            (create_segment(text="seg1"), 0.9),
            (create_segment(text="seg2"), 0.8),
        ]

        result, applied = filter.apply_face_preference(candidates)

        assert applied is False
        assert result == candidates

    def test_more_preference_boosts_high_face_score(self):
        """'more' preference should boost high face scores."""
        config = MockConfig(face_preference='more')
        filter = CandidateFilter(config=config)

        # Create segments with cached face scores (global_cache)
        seg1 = create_segment(text="seg1", source='global_cache', face_score=0.8)
        seg2 = create_segment(text="seg2", source='global_cache', face_score=0.2)

        candidates = [
            (seg1, 0.7),
            (seg2, 0.7),
        ]

        result, applied = filter.apply_face_preference(candidates)

        assert applied is True
        # seg1 with high face_score should have higher adjusted score
        scores = {c[0].text: c[1] for c in result}
        assert scores["seg1"] > scores["seg2"]

    def test_none_preference_boosts_low_face_score(self):
        """'none' preference should boost low face scores."""
        config = MockConfig(face_preference='none')
        filter = CandidateFilter(config=config)

        seg1 = create_segment(text="seg1", source='global_cache', face_score=0.8)
        seg2 = create_segment(text="seg2", source='global_cache', face_score=0.2)

        candidates = [
            (seg1, 0.7),
            (seg2, 0.7),
        ]

        result, applied = filter.apply_face_preference(candidates)

        assert applied is True
        scores = {c[0].text: c[1] for c in result}
        # seg2 with LOW face_score should have higher adjusted score for 'none' pref
        assert scores["seg2"] > scores["seg1"]


class TestLocationFilter:
    """Test location-based filtering."""

    def test_no_location_matcher_returns_unchanged(self):
        """Without location matcher, candidates should be unchanged."""
        config = MockConfig()
        filter = CandidateFilter(config=config, location_matcher=None)

        candidates = [
            (create_segment(text="seg1"), 0.9),
        ]
        vo_segment = create_segment(text="voiceover")

        result, applied, reason = filter.apply_location_filter(vo_segment, candidates)

        assert applied is False
        assert reason == ""
        assert result == candidates

    def test_delegates_to_location_matcher(self):
        """Filter should delegate to LocationMatcher when available."""
        config = MockConfig(location_enabled=True)
        mock_location_matcher = MagicMock()

        filtered_candidates = [(create_segment(text="filtered"), 0.8)]
        mock_location_matcher.apply_location_filter.return_value = (
            filtered_candidates, True, "location_match"
        )

        filter = CandidateFilter(config=config, location_matcher=mock_location_matcher)

        candidates = [(create_segment(text="seg1"), 0.9)]
        vo_segment = create_segment(text="voiceover")

        result, applied, reason = filter.apply_location_filter(vo_segment, candidates, segment_idx=5)

        assert applied is True
        assert reason == "location_match"
        assert result == filtered_candidates
        mock_location_matcher.apply_location_filter.assert_called_once()


class TestReuseFilter:
    """Test smart reuse filtering."""

    def test_filters_overused_clips(self):
        """Should filter out clips that exceed max reuse."""
        config = MockConfig()
        filter = CandidateFilter(config=config)

        seg1 = create_segment(text="seg1", source_file="video1.mp4")
        seg2 = create_segment(text="seg2", source_file="video2.mp4")

        candidates = [(seg1, 0.9), (seg2, 0.8)]

        # Create reuse tracker that blocks seg1
        reuse_tracker = MagicMock()
        reuse_tracker.can_use.side_effect = lambda seg: seg.text != "seg1"
        reuse_tracker.adjust_confidence.side_effect = lambda seg, conf: conf

        result = filter.apply_reuse_filter(candidates, reuse_tracker)

        # Only seg2 should remain
        assert len(result) == 1
        assert result[0][0].text == "seg2"

    def test_fallback_when_all_filtered(self):
        """Should use fallback with penalty when all clips filtered."""
        config = MockConfig()
        filter = CandidateFilter(config=config)

        seg1 = create_segment(text="seg1")
        candidates = [(seg1, 0.9)]

        # Reuse tracker that blocks everything
        reuse_tracker = MagicMock()
        reuse_tracker.can_use.return_value = False

        result = filter.apply_reuse_filter(candidates, reuse_tracker)

        # Should use fallback with 0.5 penalty
        assert len(result) == 1
        assert result[0][1] == 0.45  # 0.9 * 0.5


class TestApplyAllFilters:
    """Test combined filter application."""

    def test_applies_all_filters_in_order(self):
        """Should apply face, location, then reuse filters."""
        config = MockConfig(face_preference='neutral')
        filter = CandidateFilter(config=config)

        vo_segment = create_segment(text="voiceover")
        seg1 = create_segment(text="seg1")
        candidates = [(seg1, 0.9)]

        reuse_tracker = MagicMock()
        reuse_tracker.can_use.return_value = True
        reuse_tracker.adjust_confidence.side_effect = lambda seg, conf: conf

        result = filter.apply_all_filters(
            vo_segment=vo_segment,
            candidates=candidates,
            segment_idx=0,
            reuse_tracker=reuse_tracker
        )

        assert isinstance(result, FilterResult)
        assert result.original_count == 1
        assert result.filtered_count == 1
        assert result.reuse_filter_applied is True

    def test_returns_filter_metadata(self):
        """Should return metadata about which filters were applied."""
        config = MockConfig(face_preference='more')
        mock_location_matcher = MagicMock()
        mock_location_matcher.apply_location_filter.return_value = (
            [(create_segment(), 0.8)], True, "location_applied"
        )

        filter = CandidateFilter(config=config, location_matcher=mock_location_matcher)

        vo_segment = create_segment(text="voiceover")
        seg1 = create_segment(text="seg1", source='global_cache', face_score=0.5)
        candidates = [(seg1, 0.9)]

        reuse_tracker = MagicMock()
        reuse_tracker.can_use.return_value = True
        reuse_tracker.adjust_confidence.side_effect = lambda seg, conf: conf

        result = filter.apply_all_filters(
            vo_segment=vo_segment,
            candidates=candidates,
            segment_idx=0,
            reuse_tracker=reuse_tracker
        )

        assert result.face_filter_applied is True
        assert result.location_filter_applied is True
        assert result.location_reason == "location_applied"
        assert result.reuse_filter_applied is True


class TestFilterResultDataclass:
    """Test FilterResult dataclass."""

    def test_default_values(self):
        """Test FilterResult has correct defaults."""
        result = FilterResult(candidates=[])

        assert result.candidates == []
        assert result.face_filter_applied is False
        assert result.location_filter_applied is False
        assert result.location_reason == ""
        assert result.reuse_filter_applied is False
        assert result.original_count == 0
        assert result.filtered_count == 0


class TestCandidateFilterIntegration:
    """Integration tests with TieredMatcher."""

    def test_tiered_matcher_has_candidate_filter(self):
        """TieredMatcher should have CandidateFilter instance."""
        with patch('src.matching.tiered_matcher.get_config') as mock_get:
            mock_config = MockConfig()
            mock_config.matching.primary_provider = None
            mock_config.matching.secondary_provider = None
            mock_config.matching.use_local_for_review = False
            mock_config.matching.gemini_model = 'gemini-2.0-flash'
            mock_config.matching.anthropic_model = 'claude-3-haiku'
            mock_config.matching.ollama_model = 'llama3.2'
            mock_config.matching.ollama_host = 'http://localhost:11434'
            mock_config.matching.min_confidence = 0.5
            mock_config.matching.embedding_candidates = 20
            mock_config.matching.high_confidence_threshold = 0.85
            mock_config.matching.low_confidence_threshold = 0.5
            mock_config.matching.max_clip_reuse = 3
            mock_config.matching.reuse_penalty = 0.1
            mock_config.matching.chapter_matching_enabled = False
            mock_config.matching.topic_mismatch_penalty = 0.15
            mock_config.matching.ambiguous_threshold = 0.65
            mock_config.matching.cache_llm_responses = True
            mock_config.output = MagicMock()
            mock_config.output.num_alternatives = 2
            mock_config.gemini_api_key = None
            mock_config.anthropic_api_key = None
            mock_get.return_value = mock_config

            from src.matching.tiered_matcher import TieredMatcher
            matcher = TieredMatcher(config=mock_config)

            assert hasattr(matcher, 'candidate_filter')
            assert isinstance(matcher.candidate_filter, CandidateFilter)
