"""
ClaudeMatcher - Anthropic Claude LLM provider for matching.

Migrated from llm_providers.py as part of US-32-007 modularization.
Uses unified src/llm_client/ (Rule 9).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ...utils import SRTSegment
from ..llm_providers import (
    LLMProvider,
    CotReasoning,
    build_cot_batch_prompt,
    format_negative_sample_for_prompt,
    parse_cot_reasoning,
)
from src.llm_client.cost import calculate_llm_cost, get_cost_tracker

logger = logging.getLogger(__name__)


class ClaudeMatcher(LLMProvider):
    """Claude Haiku for matching (secondary/ambiguous)."""

    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        from src.llm_client import create_client
        self.client = create_client("anthropic", api_key=api_key, model=model)

    def match_batch(
        self,
        items: List[Tuple[str, List[Tuple[SRTSegment, float]]]],
        context: Optional[str] = None,
        negative_rules: Optional[List[str]] = None,
        negative_samples: Optional[List[Optional[Tuple[SRTSegment, float]]]] = None,
        use_cot: bool = False
    ) -> List[Tuple[int, float, str, Optional[CotReasoning]]]:
        from src.llm_client import LLMRequest, ResponseFormat

        if use_cot:
            prompt = build_cot_batch_prompt(items, context, negative_rules, negative_samples)
        else:
            prompt = self._build_standard_prompt(items, context, negative_rules, negative_samples)

        request = LLMRequest(
            prompt=prompt,
            response_format=ResponseFormat.JSON_ARRAY,
            cache_key_prefix="matching_cot" if use_cot else "matching",
            max_tokens=1500,
            timeout=120
        )

        try:
            response = self.client.generate(request)

            # Log LLM API call metrics (US-159-008)
            # Handle both real responses and mock objects in tests
            duration_ms = response.request_time_ms
            tokens_used = response.tokens_used
            input_tokens = response.input_tokens
            output_tokens = response.output_tokens
            provider_name = response.provider or "anthropic"
            model_name = response.model or "claude-3-haiku-20240307"
            cached = response.cached

            # Calculate cost (US-162-010)
            cost = calculate_llm_cost(
                provider=provider_name,
                model=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=tokens_used
            )

            # Track cumulative cost
            if cost > 0:
                tracker = get_cost_tracker()
                tracker.add_llm_cost(
                    cost=cost,
                    provider=provider_name,
                    input_tokens=input_tokens or 0,
                    output_tokens=output_tokens or 0,
                    total_tokens=tokens_used or 0
                )

            # Only log if values are numeric (not mock objects)
            if isinstance(duration_ms, (int, float)) and isinstance(tokens_used, (int, type(None))):
                if tokens_used and cost > 0:
                    logger.info(
                        f"LLM API call: provider={provider_name}, model={model_name}, "
                        f"duration={duration_ms:.0f}ms, tokens={tokens_used}, "
                        f"cost=${cost:.6f}, cached={cached}"
                    )
                elif tokens_used:
                    logger.info(
                        f"LLM API call: provider={provider_name}, model={model_name}, "
                        f"duration={duration_ms:.0f}ms, tokens={tokens_used}, cached={cached}"
                    )
                else:
                    logger.info(
                        f"LLM API call: provider={provider_name}, model={model_name}, "
                        f"duration={duration_ms:.0f}ms, cached={cached}"
                    )

            if response.parsed_data and isinstance(response.parsed_data, list):
                return self._process_results(response.parsed_data, items, use_cot)
            else:
                logger.warning("Claude: Could not parse response, using embedding fallback")
                raise ValueError("JSON parsing failed")
        except Exception as e:
            logger.warning(f"Claude batch error: {e}")
            raise

    def _build_standard_prompt(
        self,
        items: List[Tuple[str, List[Tuple[SRTSegment, float]]]],
        context: Optional[str],
        negative_rules: Optional[List[str]],
        negative_samples: Optional[List[Optional[Tuple[SRTSegment, float]]]]
    ) -> str:
        """Build standard (non-CoT) batch prompt."""
        batch_sections = []
        for i, (vo_text, candidates) in enumerate(items):
            vo_text_clean = vo_text.replace('"', "'")[:100]
            candidates_text = "\n".join([
                f"  {j+1}. [{Path(seg.source_file).stem[:30]}] \"{seg.text[:60].replace(chr(34), chr(39))}{'...' if len(seg.text) > 60 else ''}\""
                for j, (seg, sim) in enumerate(candidates[:5])
            ])
            negative_sample_text = ""
            if negative_samples and i < len(negative_samples) and negative_samples[i]:
                negative_sample_text = "\n" + format_negative_sample_for_prompt(negative_samples[i])
            batch_sections.append(f"VOICEOVER {i+1}: \"{vo_text_clean}\"\nCANDIDATES:\n{candidates_text}{negative_sample_text}")

        context_str = f"\nCONTEXT: {context}" if context else ""
        negative_str = ""
        if negative_rules:
            negative_str = "\n\nAVOID:\n" + "\n".join(f"- {rule}" for rule in negative_rules)
        negative_instruction = ""
        if negative_samples and any(ns is not None for ns in negative_samples):
            negative_instruction = "\nNote: Candidates marked 'unlikely match' are poor matches - do NOT select them."

        return f"""Match each voiceover to its best video candidate. Consider semantic meaning, visual relevance, and topic alignment.
{context_str}{negative_str}{negative_instruction}

{chr(10).join(batch_sections)}

Respond with ONLY a valid JSON array, no other text. Use simple reasons without special characters:
[{{"voiceover": 1, "selected": 1, "confidence": 0.85, "reason": "topic match"}}]"""

    def _process_results(
        self,
        results_list: List[Dict],
        items: List[Tuple[str, List[Tuple[SRTSegment, float]]]],
        use_cot: bool
    ) -> List[Tuple[int, float, str, Optional[CotReasoning]]]:
        """Process parsed JSON results into output tuples."""
        outputs = []
        for i, (vo_text, candidates) in enumerate(items):
            result = next((r for r in results_list if r.get('voiceover') == i + 1), None)
            if result:
                selected_idx = max(0, min(result.get('selected', 1) - 1, len(candidates) - 1))
                confidence = max(0.0, min(1.0, float(result.get('confidence', 0.7))))

                # Get the selected candidate for logging
                selected_source = candidates[selected_idx][0].source_file if selected_idx < len(candidates) else "unknown"

                cot_reasoning = None
                if use_cot:
                    cot_reasoning = parse_cot_reasoning(result)
                    if cot_reasoning.is_complete:
                        weighted_score = cot_reasoning.compute_weighted_score()
                        confidence = 0.7 * weighted_score + 0.3 * confidence

                # Log reranking decision (US-159-008)
                reason = str(result.get('reason', 'matched'))[:50]
                logger.info(
                    f"Rerank decision: vo_idx={i}, selected_idx={selected_idx}, "
                    f"source={Path(selected_source).stem[:20]}, confidence={confidence:.3f}, "
                    f"reason=\"{reason}\""
                )

                outputs.append((selected_idx, confidence, reason, cot_reasoning))
            else:
                logger.warning(f"Rerank fallback: vo_idx={i}, using embedding similarity")
                outputs.append((0, candidates[0][1] if candidates else 0.5, "parse fallback", None))
        return outputs
