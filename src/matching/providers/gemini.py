"""
GeminiMatcher - Google Gemini Flash LLM provider for matching.

Migrated from llm_providers.py as part of US-32-006 modularization.
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

logger = logging.getLogger(__name__)


class GeminiMatcher(LLMProvider):
    """Gemini Flash for matching.

    Primary LLM provider using Google's Gemini 2.0 Flash model.
    Supports batch matching with optional chain-of-thought prompting.

    Args:
        api_key: Google AI API key
        model: Gemini model name (default: gemini-2.0-flash)
    """

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        from src.llm_client import create_client
        self.client = create_client("gemini", api_key=api_key, model=model)

    def match_batch(
        self,
        items: List[Tuple[str, List[Tuple[SRTSegment, float]]]],
        context: Optional[str] = None,
        negative_rules: Optional[List[str]] = None,
        negative_samples: Optional[List[Optional[Tuple[SRTSegment, float]]]] = None,
        use_cot: bool = False
    ) -> List[Tuple[int, float, str, Optional[CotReasoning]]]:
        """
        Batch match voiceover segments to candidates.

        Args:
            items: List of (voiceover_text, candidates) tuples
            context: Optional context string for the batch
            negative_rules: Optional list of negative matching rules
            negative_samples: Optional list of negative samples (one per item), used for
                              calibrating confidence by showing what a poor match looks like
            use_cot: Whether to use chain-of-thought prompting (default: False)

        Returns:
            List of (selected_idx, confidence, reasoning, cot_reasoning) tuples.
            cot_reasoning is populated when use_cot=True, None otherwise.
        """
        from src.llm_client import LLMRequest, ResponseFormat

        # Use CoT prompt when enabled
        if use_cot:
            prompt = build_cot_batch_prompt(items, context, negative_rules, negative_samples)
            logger.info(f"    GeminiMatcher: using chain-of-thought prompt (timeout=120s)...")
        else:
            prompt = self._build_standard_prompt(items, context, negative_rules, negative_samples)
            logger.info(f"    GeminiMatcher: sending request (timeout=120s)...")

        # Call unified LLM client (handles retry and parsing)
        request = LLMRequest(
            prompt=prompt,
            response_format=ResponseFormat.JSON_ARRAY,
            cache_key_prefix="matching_cot" if use_cot else "matching",
            timeout=120
        )

        try:
            response = self.client.generate(request)
            logger.info(f"    GeminiMatcher: received response")

            # Process results
            if response.parsed_data and isinstance(response.parsed_data, list):
                return self._process_results(response.parsed_data, items, use_cot)
            else:
                logger.warning(f"Gemini: Could not parse response, using embedding fallback")
                raise ValueError("JSON parsing failed")

        except Exception as e:
            logger.warning(f"Gemini batch error: {e}")
            raise

        return [(0, 0.5, "error fallback", None) for _ in items]

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
            # Escape quotes in text to avoid JSON issues
            vo_text_clean = vo_text.replace('"', "'")
            candidates_text = "\n".join([
                f"  {j+1}. [{Path(seg.source_file).stem[:30]}] \"{seg.text[:60].replace(chr(34), chr(39))}{'...' if len(seg.text) > 60 else ''}\""
                for j, (seg, sim) in enumerate(candidates[:5])
            ])

            # Add negative sample if provided for this item
            negative_sample_text = ""
            if negative_samples and i < len(negative_samples) and negative_samples[i]:
                negative_sample_text = "\n" + format_negative_sample_for_prompt(negative_samples[i])

            batch_sections.append(f"VOICEOVER {i+1}: \"{vo_text_clean[:100]}\"\nCANDIDATES:\n{candidates_text}{negative_sample_text}")

        context_str = f"\nCONTEXT: {context}" if context else ""

        negative_str = ""
        if negative_rules:
            negative_str = "\n\nAVOID:\n" + "\n".join(f"- {rule}" for rule in negative_rules)

        # Include instruction about negative sample if any are present
        negative_instruction = ""
        if negative_samples and any(ns is not None for ns in negative_samples):
            negative_instruction = "\nNote: Candidates marked 'unlikely match' are poor matches shown for comparison - do NOT select them. Use them to calibrate your confidence scoring."

        prompt = f"""Match each voiceover to its best video candidate based on semantic meaning and topic alignment.
{context_str}{negative_str}{negative_instruction}

{chr(10).join(batch_sections)}

Respond with ONLY a valid JSON array, no other text. Use simple reasons without special characters:
[{{"voiceover": 1, "selected": 1, "confidence": 0.85, "reason": "topic match"}}]"""

        return prompt

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
                selected_idx = result.get('selected', 1) - 1
                # Clamp to valid range
                selected_idx = max(0, min(selected_idx, len(candidates) - 1))
                confidence = result.get('confidence', 0.7)
                # Clamp confidence to valid range
                confidence = max(0.0, min(1.0, float(confidence)))

                # Parse CoT reasoning if CoT is enabled
                cot_reasoning = None
                if use_cot:
                    cot_reasoning = parse_cot_reasoning(result)
                    # Use weighted score if CoT rubric is complete
                    if cot_reasoning.is_complete:
                        weighted_score = cot_reasoning.compute_weighted_score()
                        # Blend with LLM confidence (70% weighted, 30% LLM)
                        confidence = 0.7 * weighted_score + 0.3 * confidence
                        logger.debug(
                            f"    CoT: weighted={weighted_score:.3f}, blended={confidence:.3f}"
                        )

                outputs.append((
                    selected_idx,
                    confidence,
                    str(result.get('reason', 'matched'))[:50],
                    cot_reasoning
                ))
            else:
                # Fallback for missing voiceover entry
                outputs.append((0, candidates[0][1] if candidates else 0.5, "parse fallback", None))

        return outputs
