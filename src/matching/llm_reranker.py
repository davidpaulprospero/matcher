"""
LLM-based reranking of embedding search candidates.

Extracted from tiered_matcher.py for single responsibility: LLM reranking.
Selects the best video segment from candidates using LLM semantic understanding.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import logging
from typing import TYPE_CHECKING, Dict, List, Tuple, Optional, Any

if TYPE_CHECKING:
    from ..utils import SRTSegment
    from .llm_providers import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class LLMRerankerConfig:
    """Configuration for LLM reranking."""
    ambiguous_threshold: float = 0.65  # Confidence below this triggers secondary LLM
    cache_llm_responses: bool = True  # Enable response caching

    # Confidence calibration based on candidate spread (US-63-008)
    close_spread_threshold: float = 0.05  # If top-2 spread < this, apply reduction
    clear_winner_threshold: float = 0.20  # If top-2 spread > this, apply boost
    close_spread_factor: float = 0.9  # Multiply confidence by this when close spread
    clear_winner_factor: float = 1.1  # Multiply confidence by this when clear winner


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
            cache_llm_responses=getattr(matching_config, 'cache_llm_responses', True),
            close_spread_threshold=getattr(matching_config, 'llm_reranker_close_spread_threshold', 0.05),
            clear_winner_threshold=getattr(matching_config, 'llm_reranker_clear_winner_threshold', 0.20),
            close_spread_factor=getattr(matching_config, 'llm_reranker_close_spread_factor', 0.9),
            clear_winner_factor=getattr(matching_config, 'llm_reranker_clear_winner_factor', 1.1),
        )
        return cls(config, cache)

    def rerank(
        self,
        voiceover_text: str,
        candidates: List[Tuple['SRTSegment', float]],
        primary_provider: Optional['LLMProvider'] = None,
        secondary_provider: Optional['LLMProvider'] = None,
        context: Optional[str] = None,
        negative_rules: Optional[List[str]] = None,
        video_metadata: Optional[Dict[str, Dict[str, str]]] = None
    ) -> RerankResult:
        """
        Rerank candidates using LLM semantic understanding.

        Args:
            voiceover_text: Text of the voiceover segment
            candidates: List of (video_segment, similarity) tuples
            context: Optional context string
            negative_rules: Optional list of things to avoid
            video_metadata: Optional dict mapping source_file (video ID) to
                {"title": str, "description": str} for context enrichment

        Returns:
            RerankResult with selected index, confidence, and reasoning
        """
        if not candidates:
            return RerankResult(selected_idx=0, confidence=0.0, reasoning="No candidates provided")

        # Enrich candidates with video context for LLM prompt (US-70-007)
        enriched_candidates = self._enrich_candidates_with_context(
            candidates[:5], video_metadata
        )

        # Check cache first (uses enriched text for cache key)
        cache_key = self._get_cache_key(voiceover_text, enriched_candidates)
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

        # Call primary provider with enriched candidates
        try:
            results = primary_provider.match_batch(
                [(voiceover_text, enriched_candidates)],
                context=context,
                negative_rules=negative_rules
            )
            selected_idx, confidence, reasoning, _cot = results[0]

            # Check for ambiguous match - use secondary provider
            used_secondary = False
            if confidence < self.config.ambiguous_threshold and secondary_provider:
                logger.debug(f"Ambiguous match ({confidence:.2f}), using secondary LLM")
                secondary_results = secondary_provider.match_batch(
                    [(voiceover_text, enriched_candidates)],
                    context=context,
                    negative_rules=negative_rules
                )
                sec_idx, sec_conf, sec_reason, _ = secondary_results[0]

                if sec_conf > confidence:
                    selected_idx = sec_idx
                    confidence = sec_conf
                    reasoning = f"(secondary) {sec_reason}"
                    used_secondary = True

            # Apply confidence calibration based on candidate spread (US-63-008)
            confidence, spread_adjustment = self._apply_spread_calibration(
                confidence, candidates
            )
            if spread_adjustment != 0.0:
                reasoning = f"{reasoning} [spread_adj={spread_adjustment:+.2f}]"

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

    @staticmethod
    def _build_video_context(title: str, description: str) -> str:
        """Build video context string from title and description.

        Format: 'Video context: {title}. {first_sentence}'
        Returns empty string if no title/description available.
        """
        if not title and not description:
            return ""

        parts = []
        if title:
            parts.append(title)
        if description:
            # Extract first sentence (up to 100 chars)
            first_sentence = description.split('.')[0].strip()
            if len(first_sentence) > 100:
                first_sentence = first_sentence[:97] + "..."
            if first_sentence:
                parts.append(first_sentence)

        return "Video context: " + ". ".join(parts)

    def _enrich_candidates_with_context(
        self,
        candidates: List[Tuple['SRTSegment', float]],
        video_metadata: Optional[Dict[str, Dict[str, str]]] = None
    ) -> List[Tuple['SRTSegment', float]]:
        """Enrich candidate segments with video title/description context (US-70-007).

        Creates shallow copies of SRTSegments with enriched text that includes
        video context prefix when metadata is available. Falls back to original
        text when no metadata exists for a candidate.
        """
        if not video_metadata:
            return candidates

        enriched = []
        for seg, sim in candidates:
            meta = video_metadata.get(seg.source_file, {})
            title = meta.get('title', '')
            description = meta.get('description', '')
            video_context = self._build_video_context(title, description)

            if video_context:
                enriched_seg = copy.copy(seg)
                enriched_seg.text = f"[{video_context}] {seg.text}"
                enriched.append((enriched_seg, sim))
            else:
                enriched.append((seg, sim))

        return enriched

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

    def _apply_spread_calibration(
        self,
        confidence: float,
        candidates: List[Tuple['SRTSegment', float]]
    ) -> Tuple[float, float]:
        """
        Apply confidence calibration based on candidate spread (US-63-008).

        If top-2 candidates are very close (spread < close_spread_threshold),
        reduce confidence to reflect ambiguity.

        If top-2 candidates are far apart (spread > clear_winner_threshold),
        boost confidence to reflect certainty (capped at 1.0).

        Args:
            confidence: Raw confidence from LLM
            candidates: List of (segment, similarity) tuples

        Returns:
            Tuple of (calibrated_confidence, adjustment_delta)
        """
        if len(candidates) < 2:
            logger.debug("US-63-008 spread calibration: skipped (< 2 candidates)")
            return confidence, 0.0

        # Calculate spread between top-2 candidate similarities
        top_sim = candidates[0][1]
        second_sim = candidates[1][1]
        spread = top_sim - second_sim

        original_confidence = confidence
        adjustment = 0.0

        if spread < self.config.close_spread_threshold:
            # Very close candidates - reduce confidence (ambiguity)
            confidence = confidence * self.config.close_spread_factor
            adjustment = confidence - original_confidence
            logger.debug(
                f"US-63-008 spread calibration: close spread ({spread:.3f} < {self.config.close_spread_threshold}), "
                f"confidence {original_confidence:.3f} -> {confidence:.3f} (factor={self.config.close_spread_factor})"
            )
        elif spread > self.config.clear_winner_threshold:
            # Clear winner - boost confidence (capped at 1.0)
            confidence = min(1.0, confidence * self.config.clear_winner_factor)
            adjustment = confidence - original_confidence
            logger.debug(
                f"US-63-008 spread calibration: clear winner ({spread:.3f} > {self.config.clear_winner_threshold}), "
                f"confidence {original_confidence:.3f} -> {confidence:.3f} (factor={self.config.clear_winner_factor}, capped=1.0)"
            )
        else:
            logger.debug(
                f"US-63-008 spread calibration: neutral spread ({spread:.3f}), no adjustment"
            )

        return confidence, adjustment
