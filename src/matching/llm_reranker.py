"""
LLM-based reranking of embedding search candidates.

Extracted from tiered_matcher.py for single responsibility: LLM reranking.
Selects the best video segment from candidates using LLM semantic understanding.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from typing import TYPE_CHECKING, List, Tuple, Optional, Any

if TYPE_CHECKING:
    from ..utils import SRTSegment
    from .llm_providers import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class LLMRerankerConfig:
    """Configuration for LLM reranking."""
    ambiguous_threshold: float = 0.65  # Confidence below this triggers secondary LLM
    cache_llm_responses: bool = True  # Enable response caching


@dataclass
class RerankResult:
    """Result of LLM reranking."""
    selected_idx: int  # Index of selected candidate
    confidence: float  # Confidence score (0.0-1.0)
    reasoning: str  # Explanation for the selection
    used_secondary: bool = False  # Whether secondary provider was used


class LLMReranker:
    """
    LLM-based reranking for video segment selection.

    Uses LLM semantic understanding to select the best video segment
    from embedding-filtered candidates. Supports primary and secondary
    providers with automatic fallback for ambiguous matches.
    """

    def __init__(
        self,
        config: LLMRerankerConfig,
        cache: Optional[Any] = None
    ):
        """Initialize LLM reranker with config."""
        self.config = config
        self.cache = cache

    @classmethod
    def from_matching_config(
        cls,
        matching_config: Any,
        cache: Optional[Any] = None
    ) -> 'LLMReranker':
        """Create LLMReranker from matching config section."""
        config = LLMRerankerConfig(
            ambiguous_threshold=getattr(matching_config, 'ambiguous_threshold', 0.65),
            cache_llm_responses=getattr(matching_config, 'cache_llm_responses', True)
        )
        return cls(config, cache)

    def rerank(
        self,
        voiceover_text: str,
        candidates: List[Tuple['SRTSegment', float]],
        primary_provider: Optional['LLMProvider'] = None,
        secondary_provider: Optional['LLMProvider'] = None,
        context: Optional[str] = None,
        negative_rules: Optional[List[str]] = None
    ) -> RerankResult:
        """
        Rerank candidates using LLM semantic understanding.

        Args:
            voiceover_text: Text of the voiceover segment
            candidates: List of (video_segment, similarity) tuples
            context: Optional context string
            negative_rules: Optional list of things to avoid

        Returns:
            RerankResult with selected index, confidence, and reasoning
        """
        if not candidates:
            return RerankResult(selected_idx=0, confidence=0.0, reasoning="No candidates provided")

        # Check cache first
        cache_key = self._get_cache_key(voiceover_text, candidates)
        cached = self._get_cached_response(cache_key)
        if cached:
            return RerankResult(
                selected_idx=cached[0],
                confidence=cached[1],
                reasoning=f"(cached) {cached[2]}"
            )

        # No LLM provider - return embedding fallback
        if not primary_provider:
            embedding_sim = candidates[0][1] if candidates else 0.5
            return RerankResult(
                selected_idx=0,
                confidence=0.60,
                reasoning=f"Embedding similarity only (sim={embedding_sim:.2f})"
            )

        # Call primary provider
        try:
            results = primary_provider.match_batch(
                [(voiceover_text, candidates[:5])],
                context=context,
                negative_rules=negative_rules
            )
            selected_idx, confidence, reasoning, _cot = results[0]

            # Check for ambiguous match - use secondary provider
            used_secondary = False
            if confidence < self.config.ambiguous_threshold and secondary_provider:
                logger.debug(f"Ambiguous match ({confidence:.2f}), using secondary LLM")
                secondary_results = secondary_provider.match_batch(
                    [(voiceover_text, candidates[:5])],
                    context=context,
                    negative_rules=negative_rules
                )
                sec_idx, sec_conf, sec_reason, _ = secondary_results[0]

                if sec_conf > confidence:
                    selected_idx = sec_idx
                    confidence = sec_conf
                    reasoning = f"(secondary) {sec_reason}"
                    used_secondary = True

            # Cache the result
            self._cache_response(cache_key, selected_idx, confidence, reasoning)

            return RerankResult(
                selected_idx=selected_idx,
                confidence=confidence,
                reasoning=reasoning,
                used_secondary=used_secondary
            )

        except Exception as e:
            logger.warning(f"LLM reranking failed: {e}")
            embedding_sim = candidates[0][1] if candidates else 0.5
            return RerankResult(
                selected_idx=0,
                confidence=0.60,
                reasoning=f"LLM fallback (emb_sim={embedding_sim:.2f})"
            )

    def _get_cache_key(self, voiceover_text: str, candidates: List[Tuple['SRTSegment', float]]) -> str:
        """Generate cache key for LLM response."""
        content = voiceover_text + "|" + "|".join(c[0].text for c in candidates[:5])
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _get_cached_response(self, cache_key: str) -> Optional[Tuple[int, float, str]]:
        """Get cached LLM response."""
        if not self.config.cache_llm_responses or not self.cache:
            return None
        cached = self.cache.get_llm_response(cache_key)
        if cached:
            return (cached['selected'], cached['confidence'], cached['reasoning'])
        return None

    def _cache_response(self, cache_key: str, selected: int, confidence: float, reasoning: str) -> None:
        """Cache LLM response."""
        if self.config.cache_llm_responses and self.cache:
            self.cache.save_llm_response(cache_key, {
                'selected': selected,
                'confidence': confidence,
                'reasoning': reasoning
            })
