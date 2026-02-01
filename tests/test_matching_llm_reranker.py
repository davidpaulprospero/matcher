"""Tests for LLMReranker class."""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from src.matching.llm_reranker import LLMReranker, LLMRerankerConfig, RerankResult


@dataclass
class MockSRTSegment:
    """Mock SRT segment for testing."""
    text: str
    start: float = 0.0
    end: float = 1.0


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, return_values=None):
        """Initialize with optional return values."""
        self.return_values = return_values or [(0, 0.85, "Good match", "reasoning")]
        self.call_count = 0
        self.last_call_args = None

    def match_batch(self, batch, context=None, negative_rules=None):
        """Mock match_batch returning configured values."""
        self.call_count += 1
        self.last_call_args = {
            'batch': batch,
            'context': context,
            'negative_rules': negative_rules
        }
        return self.return_values


class TestLLMRerankerInit:
    """Tests for LLMReranker initialization and provider injection."""

    def test_llm_reranker_init_with_config(self):
        """Verify LLMReranker initializes with config correctly."""
        config = LLMRerankerConfig(
            ambiguous_threshold=0.7,
            cache_llm_responses=False
        )
        reranker = LLMReranker(config=config)

        assert reranker.config.ambiguous_threshold == 0.7
        assert reranker.config.cache_llm_responses is False
        assert reranker.cache is None

    def test_llm_reranker_init_with_cache(self):
        """Verify LLMReranker initializes with cache."""
        config = LLMRerankerConfig()
        mock_cache = MagicMock()
        reranker = LLMReranker(config=config, cache=mock_cache)

        assert reranker.cache is mock_cache

    def test_llm_reranker_from_matching_config(self):
        """Verify factory method creates LLMReranker from matching config."""
        matching_config = MagicMock()
        matching_config.ambiguous_threshold = 0.55
        matching_config.cache_llm_responses = True

        reranker = LLMReranker.from_matching_config(matching_config)

        assert reranker.config.ambiguous_threshold == 0.55
        assert reranker.config.cache_llm_responses is True

    def test_llm_reranker_from_matching_config_defaults(self):
        """Verify factory uses defaults when attributes missing."""
        matching_config = MagicMock(spec=[])  # Empty spec - no attributes

        reranker = LLMReranker.from_matching_config(matching_config)

        assert reranker.config.ambiguous_threshold == 0.65  # Default
        assert reranker.config.cache_llm_responses is True  # Default


class TestLLMRerankerPromptFormat:
    """Tests for LLMReranker prompt formatting via provider."""

    def test_llm_reranker_passes_voiceover_text_to_provider(self):
        """Verify voiceover text is passed to provider correctly."""
        config = LLMRerankerConfig()
        reranker = LLMReranker(config=config)
        provider = MockLLMProvider()

        candidates = [
            (MockSRTSegment("video text 1"), 0.9),
            (MockSRTSegment("video text 2"), 0.8),
        ]

        reranker.rerank(
            voiceover_text="test voiceover",
            candidates=candidates,
            primary_provider=provider
        )

        assert provider.call_count == 1
        batch = provider.last_call_args['batch']
        assert batch[0][0] == "test voiceover"

    def test_llm_reranker_passes_candidates_to_provider(self):
        """Verify candidates are passed to provider correctly."""
        config = LLMRerankerConfig()
        reranker = LLMReranker(config=config)
        provider = MockLLMProvider()

        candidates = [
            (MockSRTSegment("video text 1"), 0.9),
            (MockSRTSegment("video text 2"), 0.8),
            (MockSRTSegment("video text 3"), 0.7),
        ]

        reranker.rerank(
            voiceover_text="test voiceover",
            candidates=candidates,
            primary_provider=provider
        )

        batch = provider.last_call_args['batch']
        # Should pass up to 5 candidates
        assert len(batch[0][1]) == 3

    def test_llm_reranker_passes_context_and_negative_rules(self):
        """Verify context and negative_rules are passed through."""
        config = LLMRerankerConfig()
        reranker = LLMReranker(config=config)
        provider = MockLLMProvider()

        candidates = [(MockSRTSegment("video text"), 0.9)]

        reranker.rerank(
            voiceover_text="test voiceover",
            candidates=candidates,
            primary_provider=provider,
            context="documentary context",
            negative_rules=["no violence", "no profanity"]
        )

        assert provider.last_call_args['context'] == "documentary context"
        assert provider.last_call_args['negative_rules'] == ["no violence", "no profanity"]


class TestLLMRerankerEmptyCandidates:
    """Tests for LLMReranker handling of empty candidates."""

    def test_llm_reranker_handles_empty_candidates(self):
        """Verify empty candidates list returns valid RerankResult."""
        config = LLMRerankerConfig()
        reranker = LLMReranker(config=config)
        provider = MockLLMProvider()

        result = reranker.rerank(
            voiceover_text="test voiceover",
            candidates=[],
            primary_provider=provider
        )

        assert isinstance(result, RerankResult)
        assert result.selected_idx == 0
        assert result.confidence == 0.0
        assert "No candidates" in result.reasoning
        # Provider should not be called for empty candidates
        assert provider.call_count == 0

    def test_llm_reranker_handles_none_provider(self):
        """Verify None provider returns embedding fallback."""
        config = LLMRerankerConfig()
        reranker = LLMReranker(config=config)

        candidates = [(MockSRTSegment("video text"), 0.85)]

        result = reranker.rerank(
            voiceover_text="test voiceover",
            candidates=candidates,
            primary_provider=None
        )

        assert result.selected_idx == 0
        assert result.confidence == 0.60
        assert "Embedding similarity only" in result.reasoning


class TestLLMRerankerConfidenceThreshold:
    """Tests for LLMReranker confidence threshold behavior."""

    def test_llm_reranker_respects_confidence_threshold(self):
        """Verify ambiguous_threshold triggers secondary provider."""
        config = LLMRerankerConfig(ambiguous_threshold=0.70)
        reranker = LLMReranker(config=config)

        # Primary returns low confidence (0.5), below threshold
        primary = MockLLMProvider(return_values=[(0, 0.50, "Unsure match", "cot")])
        # Secondary returns higher confidence
        secondary = MockLLMProvider(return_values=[(1, 0.80, "Better match", "cot")])

        candidates = [
            (MockSRTSegment("video text 1"), 0.9),
            (MockSRTSegment("video text 2"), 0.8),
        ]

        result = reranker.rerank(
            voiceover_text="test voiceover",
            candidates=candidates,
            primary_provider=primary,
            secondary_provider=secondary
        )

        # Should use secondary's result since it has higher confidence
        assert result.selected_idx == 1
        assert result.confidence == 0.80
        assert result.used_secondary is True
        assert primary.call_count == 1
        assert secondary.call_count == 1

    def test_llm_reranker_skips_secondary_above_threshold(self):
        """Verify secondary not called when confidence above threshold."""
        config = LLMRerankerConfig(ambiguous_threshold=0.70)
        reranker = LLMReranker(config=config)

        # Primary returns high confidence (0.85), above threshold
        primary = MockLLMProvider(return_values=[(0, 0.85, "Good match", "cot")])
        secondary = MockLLMProvider()

        candidates = [(MockSRTSegment("video text"), 0.9)]

        result = reranker.rerank(
            voiceover_text="test voiceover",
            candidates=candidates,
            primary_provider=primary,
            secondary_provider=secondary
        )

        assert result.selected_idx == 0
        assert result.confidence == 0.85
        assert result.used_secondary is False
        assert primary.call_count == 1
        assert secondary.call_count == 0  # Not called

    def test_llm_reranker_keeps_primary_if_secondary_worse(self):
        """Verify primary result kept if secondary has lower confidence."""
        config = LLMRerankerConfig(ambiguous_threshold=0.70)
        reranker = LLMReranker(config=config)

        # Primary returns low confidence
        primary = MockLLMProvider(return_values=[(0, 0.55, "Primary match", "cot")])
        # Secondary returns even lower confidence
        secondary = MockLLMProvider(return_values=[(1, 0.40, "Worse match", "cot")])

        candidates = [
            (MockSRTSegment("video text 1"), 0.9),
            (MockSRTSegment("video text 2"), 0.8),
        ]

        result = reranker.rerank(
            voiceover_text="test voiceover",
            candidates=candidates,
            primary_provider=primary,
            secondary_provider=secondary
        )

        # Should keep primary's result
        assert result.selected_idx == 0
        assert result.confidence == 0.55
        assert result.used_secondary is False  # Secondary called but not used
