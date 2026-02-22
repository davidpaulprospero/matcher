#!/usr/bin/env python3
"""
Property-Based Tests for Scoring Module using Hypothesis

Tests mathematical properties and edge cases for scoring functions:
1. Score normalization with various input ranges
2. Confidence boost calculation edge cases
3. Temporal coherence scoring with varied inputs

Created: US-119-008
"""

import pytest
import math
from typing import List, Tuple, Optional
from unittest.mock import MagicMock

from hypothesis import given, assume, settings, example, HealthCheck
from hypothesis import strategies as st


# =============================================================================
# CUSTOM STRATEGIES
# =============================================================================

# Strategy for valid confidence scores (0.0 to 1.0)
confidence_scores = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Strategy for any float (including edge cases)
any_floats = st.floats(allow_nan=True, allow_infinity=True)

# Strategy for pool sizes
pool_sizes = st.integers(min_value=1, max_value=500)

# Strategy for small positive adjustments
small_boosts = st.floats(min_value=0.01, max_value=0.2, allow_nan=False, allow_infinity=False)

# Strategy for segments with topics/keywords
def segment_with_topics():
    """Generate mock segments with topics and keywords."""
    return st.fixed_dictionaries({
        'source_file': st.one_of(st.none(), st.text(min_size=1, max_size=100)),
        'topics': st.lists(st.text(min_size=1, max_size=30), min_size=0, max_size=10),
        'keywords': st.lists(st.text(min_size=1, max_size=30), min_size=0, max_size=10),
    })


# =============================================================================
# TESTS: SCORE NORMALIZATION
# =============================================================================

@pytest.mark.fast
class TestScoreNormalization:
    """Property tests for score normalization functions."""

    @given(
        confidence=confidence_scores,
        pool_size=pool_sizes,
    )
    @settings(max_examples=100, deadline=None)
    def test_normalize_confidence_by_pool_always_bounded(self, confidence: float, pool_size: int):
        """Normalized confidence must always be in [0, 1] range."""
        from src.matching.scoring import normalize_confidence_by_pool

        normalized, reason = normalize_confidence_by_pool(confidence, pool_size)

        assert 0.0 <= normalized <= 1.0, f"Normalized {normalized} outside [0, 1]"
        assert isinstance(reason, str), "Reason must be a string"

    @given(
        confidence=confidence_scores,
    )
    @settings(max_examples=50)
    def test_normalize_empty_pool_unchanged(self, confidence: float):
        """Empty pool (size 0) should return confidence unchanged."""
        from src.matching.scoring import normalize_confidence_by_pool

        normalized, reason = normalize_confidence_by_pool(confidence, 0)

        assert normalized == confidence, "Empty pool should return unchanged confidence"
        assert "empty_pool" in reason, "Reason should mention empty_pool"

    @given(
        confidence=confidence_scores,
    )
    @settings(max_examples=50)
    def test_normalize_disabled_returns_unchanged(self, confidence: float):
        """Disabled pool normalization should return confidence unchanged."""
        from src.matching.scoring import normalize_confidence_by_pool

        normalized, reason = normalize_confidence_by_pool(
            confidence, pool_size=100, pool_normalization_enabled=False
        )

        assert normalized == confidence, "Disabled should return unchanged confidence"
        assert "disabled" in reason.lower(), "Reason should mention disabled"

    @given(pool_size=pool_sizes)
    @settings(max_examples=50)
    def test_normalize_nan_input_returns_zero(self, pool_size: int):
        """NaN confidence input should return 0.0."""
        from src.matching.scoring import normalize_confidence_by_pool

        normalized, reason = normalize_confidence_by_pool(float('nan'), pool_size)

        assert normalized == 0.0, "NaN input should return 0.0"
        assert "nan" in reason.lower(), "Reason should mention nan"

    @given(
        confidence=st.floats(max_value=-1.0),  # Negative infinity
    )
    @settings(max_examples=10)
    def test_normalize_negative_inf_input(self, confidence: float):
        """Negative infinity input should return 0.0."""
        from src.matching.scoring import normalize_confidence_by_pool

        normalized, reason = normalize_confidence_by_pool(confidence, pool_size=10)

        assert normalized == 0.0, "Negative inf should return 0.0"

    @given(
        confidence=st.floats(min_value=1.0),  # Positive infinity
    )
    @settings(max_examples=10)
    def test_normalize_positive_inf_input(self, confidence: float):
        """Positive infinity input should return 1.0."""
        from src.matching.scoring import normalize_confidence_by_pool

        normalized, reason = normalize_confidence_by_pool(confidence, pool_size=10)

        assert normalized == 1.0, "Positive inf should return 1.0"

    @given(
        confidence=confidence_scores,
    )
    @settings(max_examples=50)
    @example(0.0)
    @example(1.0)
    @example(0.5)
    def test_normalize_extreme_confidence_values(self, confidence: float):
        """Test normalization at boundary confidence values."""
        from src.matching.scoring import normalize_confidence_by_pool

        for pool_size in [1, 10, 50, 100, 500]:
            normalized, reason = normalize_confidence_by_pool(confidence, pool_size)
            assert 0.0 <= normalized <= 1.0, f"Failed for pool_size={pool_size}"


# =============================================================================
# TESTS: CONFIDENCE BOOST CALCULATIONS
# =============================================================================

@pytest.mark.fast
class TestConfidenceBoostCalculations:
    """Property tests for confidence boost functions."""

    @given(
        confidence=confidence_scores,
        boost=small_boosts,
    )
    @settings(max_examples=100)
    def test_boost_never_exceeds_one(self, confidence: float, boost: float):
        """Boost functions should never push confidence above 1.0."""
        from src.matching.scoring import apply_broll_boost

        config = MagicMock()
        config.matching.broll_boost = boost

        segment = MagicMock()
        segment.is_broll = True

        boosted, reason = apply_broll_boost(confidence, segment, config)

        assert boosted <= 1.0, f"Boosted {boosted} exceeds 1.0 with boost={boost}"
        assert boosted >= confidence, "Boost should not decrease confidence"

    @given(
        confidence=confidence_scores,
        boost=small_boosts,
    )
    @settings(max_examples=50, deadline=None)
    def test_chapter_alignment_boost_works(self, confidence: float, boost: float):
        """Chapter alignment boost function should work and return valid boost."""
        from src.matching.scoring import compute_chapter_alignment_boost

        config = MagicMock()
        config.prefer_chapter_aligned_segments = True
        config.chapter_alignment_boost = boost

        # Mock segment with chapter info
        segment = MagicMock()
        segment.start_time = 5.0
        segment.end_time = 30.0

        # Mock chapters
        chapters = [
            {'title': 'Intro', 'start_time': 0.0, 'end_time': 10.0},
            {'title': 'Main', 'start_time': 10.0, 'end_time': 60.0},
        ]

        boost_amt, reason = compute_chapter_alignment_boost(segment, chapters, config)

        assert isinstance(boost_amt, float), "Boost amount must be float"
        assert isinstance(reason, str), "Reason must be string"
        # Boost should be non-negative
        assert boost_amt >= 0.0, "Boost should not be negative"

    @given(
        confidence=confidence_scores,
    )
    @settings(max_examples=50)
    def test_topic_alignment_boost_bounded(self, confidence: float):
        """Topic alignment boost should keep confidence in [0, 1]."""
        from src.matching.scoring import apply_topic_alignment_boost

        config = MagicMock()
        config.matching.topic_alignment_boost = 0.1
        config.matching.topic_alignment_threshold = 0.3

        vo_segment = MagicMock()
        vo_segment.topics = ["python", "tutorial"]

        video_segment = MagicMock()
        video_segment.topics = ["python", "programming"]

        boosted, reason = apply_topic_alignment_boost(
            confidence, vo_segment, video_segment, config
        )

        assert 0.0 <= boosted <= 1.0, f"Topic boost {boosted} outside [0, 1]"

    @given(
        confidence=confidence_scores,
    )
    @settings(max_examples=50)
    def test_entity_match_boost_bounded(self, confidence: float):
        """Entity match boost should keep confidence in [0, 1]."""
        from src.matching.scoring import apply_entity_match_boost

        config = MagicMock()
        config.matching.entity_match_boost = 0.15

        vo_segment = MagicMock()
        vo_segment.text = "John Smith visited Paris"

        video_segment = MagicMock()
        video_segment.text = "John Smith in Paris"

        boosted, reason, matched = apply_entity_match_boost(
            confidence, vo_segment, video_segment, config
        )

        assert 0.0 <= boosted <= 1.0, f"Entity boost {boosted} outside [0, 1]"

    @given(
        confidence=st.floats(min_value=0.9, max_value=1.0),  # High confidence
    )
    @settings(max_examples=50)
    def test_boost_at_high_confidence_caps_at_one(self, confidence: float):
        """High confidence with boost should cap at 1.0, not exceed."""
        from src.matching.scoring import apply_broll_boost

        config = MagicMock()
        config.matching.broll_boost = 0.5  # Large boost

        segment = MagicMock()
        segment.is_broll = True

        boosted, reason = apply_broll_boost(confidence, segment, config)

        assert boosted <= 1.0, f"High confidence boosted beyond 1.0: {boosted}"

    @given(
        confidence=confidence_scores,
        boost=small_boosts,
    )
    @settings(max_examples=50)
    def test_boost_never_drops_below_zero(self, confidence: float, boost: float):
        """Boost should not cause confidence to go negative (penalty case)."""
        from src.matching.scoring import apply_current_project_boost

        config = MagicMock()
        config.matching.current_project_boost = -boost  # Negative = penalty

        segment = MagicMock()
        segment.source_file = "current_project.mp4"

        penalized, reason = apply_current_project_boost(confidence, segment, config)

        assert penalized >= 0.0, f"Penalty pushed below 0: {penalized}"


# =============================================================================
# TESTS: TEMPORAL COHERENCE SCORING
# =============================================================================

@pytest.mark.fast
class TestTemporalCoherence:
    """Property tests for temporal coherence scoring."""

    @given(
        confidence=confidence_scores,
    )
    @settings(max_examples=50)
    def test_temporal_coherence_bounded(self, confidence: float):
        """Temporal coherence adjustment should keep confidence in [0, 1]."""
        from src.matching.scoring import compute_temporal_coherence

        config = MagicMock()
        config.matching.temporal_coherence_enabled = True
        config.matching.temporal_coherence_same_source_boost = 0.05
        config.matching.temporal_coherence_context_switch_penalty = 0.05

        video_segment = MagicMock()
        video_segment.source_file = "video_001.mp4"
        video_segment.topics = ["topic1"]
        video_segment.keywords = ["keyword1"]

        # Previous and next matches
        previous_match = MagicMock()
        previous_match.source_file = "video_001.mp4"  # Same source
        previous_match.topics = ["topic1"]
        previous_match.keywords = ["keyword1"]

        next_match = MagicMock()
        next_match.source_file = "video_002.mp4"  # Different source
        next_match.topics = ["topic2"]
        next_match.keywords = ["keyword2"]

        adjusted, reason = compute_temporal_coherence(
            confidence, video_segment, previous_match, next_match, config
        )

        assert 0.0 <= adjusted <= 1.0, f"Temporal coherence {adjusted} outside [0, 1]"

    @given(
        confidence=confidence_scores,
    )
    @settings(max_examples=50)
    def test_temporal_coherence_disabled_returns_unchanged(self, confidence: float):
        """Disabled temporal coherence should return confidence unchanged."""
        from src.matching.scoring import compute_temporal_coherence

        config = MagicMock()
        config.matching.temporal_coherence_enabled = False

        video_segment = MagicMock()
        video_segment.source_file = "video_001.mp4"

        adjusted, reason = compute_temporal_coherence(
            confidence, video_segment, None, None, config
        )

        assert adjusted == confidence, "Disabled temporal coherence should not change confidence"

    @given(
        confidence=confidence_scores,
    )
    @settings(max_examples=50)
    def test_temporal_coherence_no_source_file_returns_unchanged(self, confidence: float):
        """Missing source_file should return confidence unchanged."""
        from src.matching.scoring import compute_temporal_coherence

        config = MagicMock()
        config.matching.temporal_coherence_enabled = True
        config.matching.temporal_coherence_same_source_boost = 0.05

        video_segment = MagicMock()
        video_segment.source_file = None  # No source

        adjusted, reason = compute_temporal_coherence(
            confidence, video_segment, None, None, config
        )

        assert adjusted == confidence, "Missing source_file should not change confidence"

    @given(
        confidence=confidence_scores,
    )
    @settings(max_examples=50)
    def test_temporal_coherence_max_boost_bounded(self, confidence: float):
        """Maximum temporal coherence boost should be capped."""
        from src.matching.scoring import compute_temporal_coherence

        config = MagicMock()
        config.matching.temporal_coherence_enabled = True
        config.matching.temporal_coherence_same_source_boost = 0.1
        config.matching.temporal_coherence_context_switch_penalty = 0.1

        video_segment = MagicMock()
        video_segment.source_file = "video_001.mp4"
        video_segment.topics = ["topic1"]
        video_segment.keywords = ["keyword1"]

        # Both previous and next match same source -> max boost
        previous_match = MagicMock()
        previous_match.source_file = "video_001.mp4"
        previous_match.topics = ["topic1"]
        previous_match.keywords = ["keyword1"]

        next_match = MagicMock()
        next_match.source_file = "video_001.mp4"
        next_match.topics = ["topic1"]
        next_match.keywords = ["keyword1"]

        adjusted, reason = compute_temporal_coherence(
            confidence, video_segment, previous_match, next_match, config
        )

        # Max boost = 0.1 + 0.1 = 0.2
        max_expected = confidence + 0.2
        assert adjusted <= min(1.0, max_expected), f"Boost exceeded max: {adjusted}"

    @given(
        confidence=confidence_scores,
    )
    @settings(max_examples=50)
    def test_temporal_coherence_max_penalty_bounded(self, confidence: float):
        """Maximum temporal coherence penalty should be capped."""
        from src.matching.scoring import compute_temporal_coherence

        config = MagicMock()
        config.matching.temporal_coherence_enabled = True
        config.matching.temporal_coherence_same_source_boost = 0.05
        config.matching.temporal_coherence_context_switch_penalty = 0.1

        video_segment = MagicMock()
        video_segment.source_file = "video_001.mp4"
        video_segment.topics = ["topic1"]
        video_segment.keywords = ["keyword1"]

        # Both previous and next are jarring switches -> max penalty
        previous_match = MagicMock()
        previous_match.source_file = "video_999.mp4"
        previous_match.topics = ["unrelated1"]
        previous_match.keywords = ["unrelated2"]

        next_match = MagicMock()
        next_match.source_file = "video_998.mp4"
        next_match.topics = ["unrelated3"]
        next_match.keywords = ["unrelated4"]

        adjusted, reason = compute_temporal_coherence(
            confidence, video_segment, previous_match, next_match, config
        )

        # Max penalty = 0.1 + 0.1 = 0.2
        min_expected = confidence - 0.2
        assert adjusted >= max(0.0, min_expected), f"Penalty exceeded max: {adjusted}"


# =============================================================================
# TESTS: IS_JARRING_CONTEXT_SWITCH
# =============================================================================

@pytest.mark.fast
class TestJarringContextSwitch:
    """Property tests for context switch detection."""

    @given(
        topics1=st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=10),
        topics2=st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=10),
    )
    @settings(max_examples=100)
    def test_jarring_context_switch_with_overlap(self, topics1: List[str], topics2: List[str]):
        """Segments with overlapping topics should NOT be jarring."""
        from src.matching.scoring import _is_jarring_context_switch

        # Force some overlap by adding same topic to both
        if topics1 and topics2:
            shared_topic = topics1[0] if topics1 else topics2[0]
            if topics1:
                topics1[0] = shared_topic
            else:
                topics1 = [shared_topic]

        seg1 = MagicMock()
        seg1.topics = topics1
        seg1.keywords = []

        seg2 = MagicMock()
        seg2.topics = topics2
        seg2.keywords = []

        is_jarring = _is_jarring_context_switch(seg1, seg2)

        # With overlapping topics, should NOT be jarring
        if set(topics1) & set(topics2):
            assert not is_jarring, "Should not be jarring with topic overlap"

    @given(
        kw1=st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=5),
        kw2=st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=5),
    )
    @settings(max_examples=50)
    def test_jarring_with_keyword_overlap(self, kw1: List[str], kw2: List[str]):
        """Segments with overlapping keywords should NOT be jarring."""
        from src.matching.scoring import _is_jarring_context_switch

        # Force overlap
        if kw1 and kw2:
            kw1[0] = kw2[0]

        seg1 = MagicMock()
        seg1.topics = []
        seg1.keywords = kw1

        seg2 = MagicMock()
        seg2.topics = []
        seg2.keywords = kw2

        is_jarring = _is_jarring_context_switch(seg1, seg2)

        # With keyword overlap, should NOT be jarring
        if set(kw1) & set(kw2):
            assert not is_jarring, "Should not be jarring with keyword overlap"

    def test_jarring_empty_segments(self):
        """Empty segments should NOT be jarring (no content to compare)."""
        from src.matching.scoring import _is_jarring_context_switch

        seg1 = MagicMock()
        seg1.topics = []
        seg1.keywords = []

        seg2 = MagicMock()
        seg2.topics = []
        seg2.keywords = []

        is_jarring = _is_jarring_context_switch(seg1, seg2)

        assert not is_jarring, "Empty segments should not be jarring"

    def test_jarring_case_insensitive(self):
        """Topic/keyword comparison should be case insensitive."""
        from src.matching.scoring import _is_jarring_context_switch

        seg1 = MagicMock()
        seg1.topics = ["PYTHON"]
        seg1.keywords = []

        seg2 = MagicMock()
        seg2.topics = ["python"]
        seg2.keywords = []

        is_jarring = _is_jarring_context_switch(seg1, seg2)

        # Same topic (different case) should NOT be jarring
        assert not is_jarring, "Case insensitive comparison should detect overlap"


# =============================================================================
# TESTS: CHAPTER ALIGNMENT BOOST
# =============================================================================

# =============================================================================
# TESTS: POOL BATCH NORMALIZATION
# =============================================================================

@pytest.mark.fast
class TestPoolBatchNormalization:
    """Property tests for batch pool normalization."""

    @given(
        conf1=confidence_scores,
        conf2=confidence_scores,
        conf3=confidence_scores,
    )
    @settings(max_examples=50, deadline=None)
    def test_normalize_pool_batch_bounded(self, conf1: float, conf2: float, conf3: float):
        """Batch normalization should keep all scores in [0, 1]."""
        from src.matching.scoring import normalize_pool_batch

        # Each segment is (confidence, pool_size, candidates)
        segments = [
            (conf, 50, None)
            for conf in [conf1, conf2, conf3]
        ]

        results = normalize_pool_batch(segments)

        assert len(results) == 3, "Should return result for each segment"
        for i, (normalized, reason) in enumerate(results):
            assert 0.0 <= normalized <= 1.0, f"Result {i} outside [0, 1]: {normalized}"
            assert isinstance(reason, str), f"Reason {i} not a string"

    def test_normalize_pool_batch_empty_segments(self):
        """Batch normalization should handle empty segments list."""
        from src.matching.scoring import normalize_pool_batch

        results = normalize_pool_batch([])

        assert results == [], "Empty segments should return empty results"


# =============================================================================
# EDGE CASE EXAMPLES
# =============================================================================

@pytest.mark.fast
class TestScoringEdgeCases:
    """Explicit edge case tests for scoring functions."""

    @given(st.integers(min_value=1, max_value=1000))
    @example(1)
    @example(10)
    @example(50)
    @example(100)
    @example(500)
    @example(1000)
    def test_normalize_at_various_pool_sizes(self, pool_size: int):
        """Test normalization at various pool sizes."""
        from src.matching.scoring import normalize_confidence_by_pool

        confidence = 0.8
        normalized, reason = normalize_confidence_by_pool(confidence, pool_size)

        assert 0.0 <= normalized <= 1.0, f"Failed at pool_size={pool_size}"

    @given(confidence_scores)
    @example(0.0)
    @example(1.0)
    @example(0.001)
    @example(0.999)
    def test_boost_at_confidence_boundaries(self, confidence: float):
        """Test boost functions at confidence boundaries."""
        from src.matching.scoring import apply_broll_boost

        config = MagicMock()
        config.matching.broll_boost = 0.1

        segment = MagicMock()
        segment.is_broll = True

        boosted, reason = apply_broll_boost(confidence, segment, config)

        assert 0.0 <= boosted <= 1.0, f"Failed at confidence={confidence}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
