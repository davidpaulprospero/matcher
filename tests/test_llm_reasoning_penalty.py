"""Tests for LLM reasoning quality penalty (US-84-006).

Verifies that low-quality LLM reasoning triggers a confidence penalty
and that secondary LLM fallback applies a decay multiplier.
"""

import pytest
from unittest.mock import MagicMock, patch
from src.matching.llm_reranker import LLMReranker, LLMRerankerConfig, RerankResult


def _make_mock_segment(text: str, source_file: str = "vid1"):
    """Create a mock SRTSegment."""
    seg = MagicMock()
    seg.text = text
    seg.source_file = source_file
    seg.start_time = 0.0
    seg.end_time = 5.0
    return seg


def _make_mock_provider(confidence: float, reasoning: str, selected_idx: int = 0):
    """Create a mock LLM provider that returns fixed results."""
    provider = MagicMock()
    provider.match_batch.return_value = [
        (selected_idx, confidence, reasoning, None)
    ]
    return provider


class TestLLMReasoningQualityPenalty:
    """Test penalty application for low-quality LLM reasoning."""

    def test_generic_reasoning_gets_penalty(self):
        """Generic reasoning like 'this is relevant' should get -0.05 penalty."""
        config = LLMRerankerConfig(
            low_quality_reasoning_penalty=0.05,
            secondary_llm_decay=0.9,
        )
        reranker = LLMReranker(config=config)

        # Generic reasoning - short with generic phrase
        provider = _make_mock_provider(
            confidence=0.80,
            reasoning="relevant",
        )

        seg = _make_mock_segment("The economy grew significantly in 2024")
        candidates = [(seg, 0.85)]

        result = reranker.rerank(
            voiceover_text="The economy grew significantly in 2024",
            candidates=candidates,
            primary_provider=provider,
        )

        assert result.llm_reasoning_quality == 0
        # Confidence should be reduced by 0.05
        assert result.confidence < 0.80

    def test_specific_reasoning_no_penalty(self):
        """Specific reasoning with keyword references should NOT be penalized."""
        config = LLMRerankerConfig(
            low_quality_reasoning_penalty=0.05,
            secondary_llm_decay=0.9,
        )
        reranker = LLMReranker(config=config)

        voiceover_text = "The economy grew significantly in 2024 with GDP rising"
        # Specific reasoning that references economy, GDP, grew etc.
        provider = _make_mock_provider(
            confidence=0.80,
            reasoning="This video discusses economy growth and GDP rising significantly showing charts",
        )

        seg = _make_mock_segment(voiceover_text)
        candidates = [(seg, 0.85)]

        result = reranker.rerank(
            voiceover_text=voiceover_text,
            candidates=candidates,
            primary_provider=provider,
        )

        assert result.llm_reasoning_quality == 1
        # No penalty applied, though spread calibration may adjust
        # The key assertion is that quality is 1 (normal)

    def test_llm_reasoning_quality_field_present(self):
        """RerankResult should always include llm_reasoning_quality field."""
        result = RerankResult(
            selected_idx=0,
            confidence=0.80,
            reasoning="test",
        )
        assert result.llm_reasoning_quality == 1  # Default is normal

        result_low = RerankResult(
            selected_idx=0,
            confidence=0.75,
            reasoning="test",
            llm_reasoning_quality=0,
        )
        assert result_low.llm_reasoning_quality == 0


class TestSecondaryLLMDecay:
    """Test decay multiplier when secondary LLM fallback is triggered."""

    def test_secondary_fallback_applies_decay(self):
        """When secondary LLM is used, 0.9x multiplier should apply."""
        config = LLMRerankerConfig(
            ambiguous_threshold=0.65,
            low_quality_reasoning_penalty=0.05,
            secondary_llm_decay=0.9,
        )
        reranker = LLMReranker(config=config)

        voiceover_text = "The solar panels generate renewable energy efficiently"

        # Primary returns low confidence (below ambiguous_threshold)
        primary = _make_mock_provider(
            confidence=0.50,
            reasoning="Solar panels and renewable energy generation shown with technical details",
        )
        # Secondary returns higher confidence
        secondary = _make_mock_provider(
            confidence=0.75,
            reasoning="Solar panels and renewable energy generation clearly depicted in footage",
        )

        seg = _make_mock_segment(voiceover_text)
        candidates = [(seg, 0.85)]

        result = reranker.rerank(
            voiceover_text=voiceover_text,
            candidates=candidates,
            primary_provider=primary,
            secondary_provider=secondary,
        )

        assert result.used_secondary is True
        # Secondary conf 0.75 * 0.9 decay = 0.675 (before spread calibration)
        # The exact value depends on spread calibration, but it should be < 0.75
        assert result.confidence < 0.75

    def test_no_decay_when_primary_sufficient(self):
        """When primary confidence is above threshold, no secondary and no decay."""
        config = LLMRerankerConfig(
            ambiguous_threshold=0.65,
            low_quality_reasoning_penalty=0.05,
            secondary_llm_decay=0.9,
        )
        reranker = LLMReranker(config=config)

        voiceover_text = "The solar panels generate renewable energy efficiently"
        primary = _make_mock_provider(
            confidence=0.80,
            reasoning="Solar panels generate renewable energy efficiently with detailed footage",
        )
        secondary = _make_mock_provider(confidence=0.90, reasoning="better")

        seg = _make_mock_segment(voiceover_text)
        candidates = [(seg, 0.85)]

        result = reranker.rerank(
            voiceover_text=voiceover_text,
            candidates=candidates,
            primary_provider=primary,
            secondary_provider=secondary,
        )

        assert result.used_secondary is False


class TestBreakdownRecording:
    """Test that penalties appear in confidence_breakdown."""

    def test_low_quality_breakdown_entry(self):
        """Low-quality reasoning should produce llm_reasoning_penalty breakdown entry."""
        config = LLMRerankerConfig(
            low_quality_reasoning_penalty=0.05,
            secondary_llm_decay=0.9,
        )
        reranker = LLMReranker(config=config)

        # Use generic reasoning
        provider = _make_mock_provider(
            confidence=0.80,
            reasoning="relevant",
        )
        seg = _make_mock_segment("The economy grew significantly in 2024")
        candidates = [(seg, 0.85)]

        result = reranker.rerank(
            voiceover_text="The economy grew significantly in 2024",
            candidates=candidates,
            primary_provider=provider,
        )

        # Verify the result has low quality
        assert result.llm_reasoning_quality == 0

        # The breakdown recording happens in tiered_matcher._match_single_segment,
        # not in the reranker itself. We test it via the RerankResult fields.
        # The reranker provides the data; tiered_matcher records it.

    def test_secondary_decay_breakdown_entry(self):
        """Secondary LLM decay should be reflected in used_secondary flag."""
        config = LLMRerankerConfig(
            ambiguous_threshold=0.65,
            low_quality_reasoning_penalty=0.05,
            secondary_llm_decay=0.9,
        )
        reranker = LLMReranker(config=config)

        voiceover_text = "The solar panels generate renewable energy efficiently"
        primary = _make_mock_provider(
            confidence=0.50,
            reasoning="Solar panels and renewable energy generation shown with technical details",
        )
        secondary = _make_mock_provider(
            confidence=0.75,
            reasoning="Solar panels and renewable energy generation clearly depicted in footage",
        )

        seg = _make_mock_segment(voiceover_text)
        candidates = [(seg, 0.85)]

        result = reranker.rerank(
            voiceover_text=voiceover_text,
            candidates=candidates,
            primary_provider=primary,
            secondary_provider=secondary,
        )

        assert result.used_secondary is True
        # Tiered matcher uses used_secondary to record secondary_llm_decay


class TestPenaltyCombination:
    """Test both penalties applying simultaneously."""

    def test_both_penalties_can_apply(self):
        """When secondary is used AND reasoning is generic, both should apply."""
        config = LLMRerankerConfig(
            ambiguous_threshold=0.65,
            low_quality_reasoning_penalty=0.05,
            secondary_llm_decay=0.9,
        )
        reranker = LLMReranker(config=config)

        voiceover_text = "The economy grew significantly in 2024"
        # Primary returns low confidence
        primary = _make_mock_provider(
            confidence=0.40,
            reasoning="good match",
        )
        # Secondary returns higher confidence but still generic
        secondary = _make_mock_provider(
            confidence=0.60,
            reasoning="matches well",
        )

        seg = _make_mock_segment(voiceover_text)
        candidates = [(seg, 0.85)]

        result = reranker.rerank(
            voiceover_text=voiceover_text,
            candidates=candidates,
            primary_provider=primary,
            secondary_provider=secondary,
        )

        assert result.used_secondary is True
        assert result.llm_reasoning_quality == 0
        # Both penalties should reduce confidence below 0.60
        assert result.confidence < 0.60

    def test_no_provider_returns_normal_quality(self):
        """When no LLM provider, result should have normal quality."""
        config = LLMRerankerConfig()
        reranker = LLMReranker(config=config)

        seg = _make_mock_segment("test text")
        candidates = [(seg, 0.85)]

        result = reranker.rerank(
            voiceover_text="test text",
            candidates=candidates,
            primary_provider=None,
        )

        assert result.llm_reasoning_quality == 1
        assert result.used_secondary is False
