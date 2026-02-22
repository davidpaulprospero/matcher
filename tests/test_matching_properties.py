#!/usr/bin/env python3
"""
Property-Based Tests for Matching Module

Uses hypothesis to generate test cases and verify mathematical properties
that must hold for all inputs. This catches edge cases that example-based
tests miss.

Properties tested:
1. Confidence scores: Always in [0, 1] range
2. Embedding similarity: Cosine similarity properties
3. Match ranking: Deterministic and monotonic

Created: Sprint 27 (US-005)
"""

import pytest
import math
from typing import List, Tuple, Optional
from unittest.mock import MagicMock

import numpy as np
from hypothesis import given, assume, settings, example, HealthCheck
from hypothesis import strategies as st


# =============================================================================
# CUSTOM STRATEGIES
# =============================================================================

# Strategy for valid confidence scores (0.0 to 1.0)
confidence_scores = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Strategy for valid thresholds (typically 0.5 to 0.99)
thresholds = st.floats(min_value=0.5, max_value=0.99, allow_nan=False, allow_infinity=False)

# Strategy for penalty/boost factors (small positive values)
small_factors = st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False)

# Strategy for embedding vectors (normalized or unnormalized)
embedding_dims = st.integers(min_value=1, max_value=512)

def embedding_vector(dim: int = 384) -> st.SearchStrategy:
    """Generate an embedding vector of fixed dimension."""
    return st.lists(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=dim,
        max_size=dim
    )

# Strategy for normalized embedding vectors (unit length)
def normalized_embedding(dim: int = 384) -> st.SearchStrategy:
    """Generate a unit-norm embedding vector."""
    return embedding_vector(dim).map(lambda v: _normalize(v) if sum(x*x for x in v) > 0 else [1.0] + [0.0]*(dim-1))

def _normalize(v: List[float]) -> List[float]:
    """Normalize a vector to unit length."""
    norm = math.sqrt(sum(x*x for x in v))
    if norm == 0:
        return v
    return [x / norm for x in v]

# Strategy for voiceover text
voiceover_text = st.text(min_size=0, max_size=500, alphabet=st.characters(blacklist_categories=('Cs',)))

# Strategy for keyword lists
keyword_lists = st.lists(
    st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N'))),
    min_size=0,
    max_size=20
)

# Strategy for pool sizes
pool_sizes = st.integers(min_value=1, max_value=500)

# Strategy for speed ratios (video/voiceover duration)
speed_ratios = st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False)

# Strategy for quality weights dict
quality_weights = st.fixed_dictionaries({
    'text_embedding': st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    'keyword_overlap': st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    'entity_match': st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    'visual_description': st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
})


# =============================================================================
# PROPERTY TESTS: CONFIDENCE SCORE CALCULATIONS
# =============================================================================

@pytest.mark.fast
class TestConfidenceScoreProperties:
    """Property tests for confidence score functions."""

    @given(
        base_threshold=thresholds,
        vo_text=voiceover_text,
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_adaptive_threshold_stays_in_valid_range(self, base_threshold: float, vo_text: str):
        """Adaptive threshold must always be in [0.5, 0.99] range."""
        from src.matching.scoring import calculate_adaptive_threshold

        # Create mock candidates with various similarity scores
        candidates = [(MagicMock(), 0.7), (MagicMock(), 0.6), (MagicMock(), 0.5)]

        adjusted, reason = calculate_adaptive_threshold(base_threshold, vo_text, candidates)

        assert 0.5 <= adjusted <= 0.99, f"Threshold {adjusted} outside valid range [0.5, 0.99]"
        assert isinstance(reason, str), "Reason must be a string"

    @given(
        confidence=confidence_scores,
        broll_boost=small_factors,
    )
    @settings(max_examples=100)
    def test_broll_boost_never_exceeds_one(self, confidence: float, broll_boost: float):
        """B-roll boost must never push confidence above 1.0."""
        from src.matching.scoring import apply_broll_boost

        # Create mock config
        config = MagicMock()
        config.matching.broll_boost = broll_boost

        # Create mock segment marked as B-roll
        segment = MagicMock()
        segment.is_broll = True

        boosted, reason = apply_broll_boost(confidence, segment, config)

        assert 0.0 <= boosted <= 1.0, f"Boosted confidence {boosted} outside [0, 1]"
        assert boosted >= confidence, f"B-roll boost should never decrease confidence"

    @given(
        confidence=confidence_scores,
        quality_tier=st.sampled_from(['high', 'medium', 'low']),
    )
    @settings(max_examples=100)
    def test_caption_quality_adjustment_bounded(self, confidence: float, quality_tier: str):
        """Caption quality adjustment must keep confidence in [0, 1]."""
        from src.matching.scoring import apply_caption_quality_adjustment

        config = MagicMock()
        config.matching.caption_quality_adjustment_enabled = True
        config.matching.caption_quality_weights = None
        config.matching.caption_quality_high_boost = 0.05
        config.matching.caption_quality_low_penalty = 0.1

        segment = MagicMock()
        segment.caption_quality = quality_tier

        adjusted, reason = apply_caption_quality_adjustment(confidence, segment, config)

        assert 0.0 <= adjusted <= 1.0, f"Adjusted confidence {adjusted} outside [0, 1]"

    @given(
        confidence=confidence_scores,
        timing_penalty=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_timing_penalty_monotonic(self, confidence: float, timing_penalty: float):
        """Higher timing penalty (closer to 0) should always reduce confidence more."""
        from src.matching.scoring import apply_timing_penalty

        config = MagicMock()
        config.matching.apply_timing_penalty = True

        segment = MagicMock()
        segment.timing_penalty = timing_penalty

        adjusted, reason = apply_timing_penalty(confidence, segment, config)

        assert 0.0 <= adjusted <= 1.0, f"Adjusted confidence {adjusted} outside [0, 1]"
        # With penalty < 1.0, result should be <= original
        if timing_penalty < 1.0:
            assert adjusted <= confidence + 0.001, "Timing penalty should not increase confidence"

    @given(
        confidence=confidence_scores,
        pool_size=pool_sizes,
    )
    @settings(max_examples=100)
    def test_pool_normalization_bounded(self, confidence: float, pool_size: int):
        """Pool normalization must keep confidence in [0, 1]."""
        from src.matching.scoring import normalize_confidence_by_pool

        normalized, reason = normalize_confidence_by_pool(confidence, pool_size)

        assert 0.0 <= normalized <= 1.0, f"Normalized confidence {normalized} outside [0, 1]"
        assert isinstance(reason, str), "Reason must be a string"


# =============================================================================
# PROPERTY TESTS: EMBEDDING SIMILARITY
# =============================================================================

@pytest.mark.fast
class TestEmbeddingSimilarityProperties:
    """Property tests for embedding similarity functions."""

    @given(
        v1=st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=8, max_size=8),
        v2=st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=8, max_size=8),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_cosine_similarity_bounded(self, v1: List[float], v2: List[float]):
        """Cosine similarity must be in [-1, 1] range."""
        from src.embeddings import _compute_cosine_similarity

        arr1 = np.array(v1, dtype=np.float32)
        arr2 = np.array(v2, dtype=np.float32)

        # Skip zero vectors
        if np.linalg.norm(arr1) == 0 or np.linalg.norm(arr2) == 0:
            return

        similarity = _compute_cosine_similarity(arr1, arr2)

        assert -1.0 <= similarity <= 1.0 + 1e-6, f"Cosine similarity {similarity} outside [-1, 1]"

    @given(
        v=st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=8, max_size=8),
    )
    @settings(max_examples=50)
    def test_cosine_similarity_self_is_one(self, v: List[float]):
        """Cosine similarity of a vector with itself must be 1.0."""
        from src.embeddings import _compute_cosine_similarity

        arr = np.array(v, dtype=np.float32)

        # Skip zero vectors
        if np.linalg.norm(arr) == 0:
            return

        similarity = _compute_cosine_similarity(arr, arr)

        assert abs(similarity - 1.0) < 1e-5, f"Self-similarity {similarity} should be ~1.0"

    @given(
        v1=st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=8, max_size=8),
        v2=st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=8, max_size=8),
    )
    @settings(max_examples=50)
    def test_cosine_similarity_symmetric(self, v1: List[float], v2: List[float]):
        """Cosine similarity must be symmetric: sim(a, b) == sim(b, a)."""
        from src.embeddings import _compute_cosine_similarity

        arr1 = np.array(v1, dtype=np.float32)
        arr2 = np.array(v2, dtype=np.float32)

        # Skip zero vectors
        if np.linalg.norm(arr1) == 0 or np.linalg.norm(arr2) == 0:
            return

        sim_ab = _compute_cosine_similarity(arr1, arr2)
        sim_ba = _compute_cosine_similarity(arr2, arr1)

        assert abs(sim_ab - sim_ba) < 1e-5, f"Asymmetric: sim(a,b)={sim_ab}, sim(b,a)={sim_ba}"


# =============================================================================
# PROPERTY TESTS: MATCH RANKING ALGORITHMS
# =============================================================================

@pytest.mark.fast
class TestMatchRankingProperties:
    """Property tests for match ranking functions."""

    @given(
        vo_keywords=keyword_lists,
        video_keywords=keyword_lists,
    )
    @settings(max_examples=100)
    def test_keyword_overlap_score_bounded(self, vo_keywords: List[str], video_keywords: List[str]):
        """Keyword overlap score must be in [0, 1] range."""
        from src.matching.scoring import calculate_keyword_overlap_score

        score, matched = calculate_keyword_overlap_score(vo_keywords, video_keywords)

        assert 0.0 <= score <= 1.0, f"Keyword overlap score {score} outside [0, 1]"
        assert isinstance(matched, list), "Matched keywords must be a list"
        # If there are matches, score should be positive
        if matched:
            assert score > 0, "Score should be positive when there are matches"

    @given(
        vo_keywords=keyword_lists,
        video_keywords=keyword_lists,
    )
    @settings(max_examples=50)
    def test_keyword_overlap_symmetric(self, vo_keywords: List[str], video_keywords: List[str]):
        """Keyword overlap should be symmetric (same matches either direction)."""
        from src.matching.scoring import calculate_keyword_overlap_score

        score1, matched1 = calculate_keyword_overlap_score(vo_keywords, video_keywords)
        score2, matched2 = calculate_keyword_overlap_score(video_keywords, vo_keywords)

        # Matched sets should be equivalent (though order may differ)
        set1 = {k.lower() for k in matched1}
        set2 = {k.lower() for k in matched2}
        assert set1 == set2, f"Asymmetric matches: {matched1} vs {matched2}"

    @given(
        vo_entities=keyword_lists,
        video_entities=keyword_lists,
    )
    @settings(max_examples=100)
    def test_entity_match_score_bounded(self, vo_entities: List[str], video_entities: List[str]):
        """Entity match score must be in [0, 1] range."""
        from src.matching.scoring import calculate_entity_match_score

        score, matched = calculate_entity_match_score(vo_entities, video_entities)

        assert 0.0 <= score <= 1.0, f"Entity match score {score} outside [0, 1]"
        # If there are matches, score should be positive
        if matched:
            assert score > 0, "Score should be positive when there are matches"

    @given(
        emb_sim=confidence_scores,
        kw_overlap=confidence_scores,
        entity_match=confidence_scores,
        visual_desc=confidence_scores,
    )
    @settings(max_examples=100)
    def test_multimodal_score_bounded(
        self, emb_sim: float, kw_overlap: float, entity_match: float, visual_desc: float
    ):
        """Multimodal score must be in [0, 1] range."""
        from src.matching.scoring import compute_multimodal_score

        score, reason, components = compute_multimodal_score(
            emb_sim, kw_overlap, entity_match, visual_desc
        )

        assert 0.0 <= score <= 1.0, f"Multimodal score {score} outside [0, 1]"
        assert isinstance(reason, str), "Reason must be a string"
        assert isinstance(components, dict), "Components must be a dict"

    @given(
        emb_sim=confidence_scores,
        kw_overlap=confidence_scores,
        entity_match=confidence_scores,
        visual_desc=confidence_scores,
        weights=quality_weights,
    )
    @settings(max_examples=50)
    def test_multimodal_score_with_custom_weights(
        self, emb_sim: float, kw_overlap: float, entity_match: float,
        visual_desc: float, weights: dict
    ):
        """Multimodal score with custom weights must be bounded and handle zero weights."""
        from src.matching.scoring import compute_multimodal_score

        # Test doesn't crash with any weight combination
        score, reason, components = compute_multimodal_score(
            emb_sim, kw_overlap, entity_match, visual_desc,
            weights=weights, multimodal_enabled=True
        )

        assert 0.0 <= score <= 1.0, f"Multimodal score {score} outside [0, 1]"


# =============================================================================
# PROPERTY TESTS: TRANSCRIPT QUALITY
# =============================================================================

@pytest.mark.fast
class TestTranscriptQualityProperties:
    """Property tests for transcript quality scoring."""

    @given(
        text=st.text(min_size=0, max_size=2000, alphabet=st.characters(blacklist_categories=('Cs',))),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_transcript_quality_bounded(self, text: str):
        """Transcript quality score must be in [0, 1] range."""
        from src.matching.scoring import calculate_transcript_quality

        score, tier, reason = calculate_transcript_quality(text)

        assert 0.0 <= score <= 1.0, f"Quality score {score} outside [0, 1]"
        assert tier in ('high', 'medium', 'low'), f"Invalid tier: {tier}"
        assert isinstance(reason, str), "Reason must be a string"

    @given(
        text=st.text(min_size=0, max_size=2000, alphabet=st.characters(blacklist_categories=('Cs',))),
    )
    @settings(max_examples=50)
    def test_transcript_quality_tier_consistent_with_score(self, text: str):
        """Quality tier must be consistent with score thresholds."""
        from src.matching.scoring import (
            calculate_transcript_quality,
            TRANSCRIPT_QUALITY_HIGH_THRESHOLD,
            TRANSCRIPT_QUALITY_MEDIUM_THRESHOLD,
        )

        score, tier, reason = calculate_transcript_quality(text)

        if score >= TRANSCRIPT_QUALITY_HIGH_THRESHOLD:
            assert tier == 'high', f"Score {score} should be 'high' tier"
        elif score >= TRANSCRIPT_QUALITY_MEDIUM_THRESHOLD:
            assert tier == 'medium', f"Score {score} should be 'medium' tier"
        else:
            assert tier == 'low', f"Score {score} should be 'low' tier"


# =============================================================================
# PROPERTY TESTS: DURATION SCORING
# =============================================================================

@pytest.mark.fast
class TestDurationScoringProperties:
    """Property tests for duration-based scoring."""

    @given(
        confidence=confidence_scores,
        speed_ratio=speed_ratios,
        penalty_factor=small_factors,
    )
    @settings(max_examples=100)
    def test_duration_penalty_bounded_adjustment(
        self, confidence: float, speed_ratio: float, penalty_factor: float
    ):
        """Duration adjustment should be bounded: max boost is reward_boost, max penalty is 2*factor."""
        from src.matching.scoring import apply_duration_penalty

        config = MagicMock()
        config.matching.duration_penalty_factor = penalty_factor
        scoring_mock = MagicMock()
        scoring_mock.duration_ratio_reward_threshold = 0.1
        scoring_mock.duration_ratio_reward_boost = 0.02
        config.matching.scoring = scoring_mock

        adjusted = apply_duration_penalty(confidence, speed_ratio, config)

        # Max boost is reward_boost (0.02), max penalty is penalty_factor * 2
        assert adjusted <= confidence + 0.021, "Adjustment should not exceed reward boost"
        assert adjusted >= confidence - (penalty_factor * 2) - 0.001, "Penalty should not exceed 2x factor"

    @given(
        confidence=confidence_scores,
        speed_ratio=st.floats(min_value=0.9, max_value=1.1, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_near_perfect_ratio_gets_reward(self, confidence: float, speed_ratio: float):
        """Speed ratios within reward threshold get a boost (US-84-007)."""
        from src.matching.scoring import apply_duration_penalty

        config = MagicMock()
        config.matching.duration_penalty_factor = 0.1
        scoring_mock = MagicMock()
        scoring_mock.duration_ratio_reward_threshold = 0.1
        scoring_mock.duration_ratio_reward_boost = 0.02
        config.matching.scoring = scoring_mock

        adjusted = apply_duration_penalty(confidence, speed_ratio, config)

        assert adjusted == pytest.approx(confidence + 0.02, abs=1e-4), "Near-perfect ratio should get reward boost"


# =============================================================================
# PROPERTY TESTS: SEMANTIC COHERENCE
# =============================================================================

@pytest.mark.fast
class TestSemanticCoherenceProperties:
    """Property tests for semantic coherence scoring."""

    @given(
        v1=st.lists(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False), min_size=8, max_size=8),
        v2=st.lists(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False), min_size=8, max_size=8),
    )
    @settings(max_examples=50)
    def test_semantic_coherence_adjustment_bounded(self, v1: List[float], v2: List[float]):
        """Semantic coherence adjustment should be small and bounded."""
        from src.matching.scoring import compute_semantic_coherence

        arr1 = np.array(v1, dtype=np.float32)
        arr2 = np.array(v2, dtype=np.float32)

        # Skip zero vectors
        if np.linalg.norm(arr1) == 0 or np.linalg.norm(arr2) == 0:
            return

        adjustment, reason = compute_semantic_coherence(arr1, arr2)

        # Adjustment should be bounded to small values
        assert -0.1 <= adjustment <= 0.1, f"Adjustment {adjustment} too large"
        assert isinstance(reason, str), "Reason must be a string"


# =============================================================================
# EDGE CASE EXAMPLES
# =============================================================================

@pytest.mark.fast
class TestEdgeCaseExamples:
    """Explicit edge case tests using hypothesis examples."""

    @given(confidence_scores)
    @example(0.0)  # Minimum
    @example(1.0)  # Maximum
    @example(0.5)  # Middle
    def test_confidence_edge_cases(self, confidence: float):
        """Test confidence scoring at boundary values."""
        from src.matching.scoring import apply_broll_boost

        config = MagicMock()
        config.matching.broll_boost = 0.1

        segment = MagicMock()
        segment.is_broll = True

        boosted, _ = apply_broll_boost(confidence, segment, config)
        assert 0.0 <= boosted <= 1.0

    @given(st.integers(min_value=0, max_value=1000))
    @example(0)  # Empty pool edge case - should return confidence unchanged
    @example(1)  # Single candidate
    @example(50)  # Reference size
    @example(500)  # Large pool
    def test_pool_size_edge_cases(self, pool_size: int):
        """Test pool normalization at boundary pool sizes."""
        from src.matching.scoring import normalize_confidence_by_pool

        confidence = 0.8

        if pool_size == 0:
            # Empty pool should return unchanged
            normalized, reason = normalize_confidence_by_pool(confidence, pool_size)
            assert normalized == confidence
            assert "empty_pool" in reason
        else:
            normalized, reason = normalize_confidence_by_pool(confidence, pool_size)
            assert 0.0 <= normalized <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
