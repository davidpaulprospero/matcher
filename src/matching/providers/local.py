"""
Local LLM (Ollama) provider for matching.

Extracted from llm_providers.py to reduce file size.
Uses unified src/llm_client/ (Rule 9).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ...utils import SRTSegment
from ..llm_providers import LLMProvider, CotReasoning, parse_cot_reasoning

logger = logging.getLogger(__name__)


class LocalLLMMatcher(LLMProvider):
    """Local LLM (Ollama) for finishing touches"""

    def __init__(self, model: str = "llama3.2", host: str = "http://localhost:11434"):
        from src.llm_client import create_client
        self.client = create_client("ollama", model=model, host=host)

    def match_batch(
        self,
        items: List[Tuple[str, List[Tuple[SRTSegment, float]]]],
        context: Optional[str] = None,
        negative_rules: Optional[List[str]] = None,
        negative_samples: Optional[List[Optional[Tuple[SRTSegment, float]]]] = None,
        use_cot: bool = False
    ) -> List[Tuple[int, float, str, Optional[CotReasoning]]]:
        from src.llm_client import LLMRequest, ResponseFormat

        outputs = []

        # Process items one at a time (Ollama works better this way)
        for idx, (vo_text, candidates) in enumerate(items):
            # Escape quotes to avoid JSON issues
            vo_text_clean = vo_text.replace('"', "'")[:100]
            candidates_text = "\n".join([
                f"{j+1}. \"{seg.text[:80].replace(chr(34), chr(39))}\""
                for j, (seg, sim) in enumerate(candidates[:5])
            ])

            # Add negative sample if provided
            negative_sample_text = ""
            if negative_samples and idx < len(negative_samples) and negative_samples[idx]:
                neg_seg, neg_sim = negative_samples[idx]
                neg_text = neg_seg.text[:80].replace('"', "'") if neg_seg.text else ""
                negative_sample_text = f"\nUNLIKELY. \"{neg_text}\" (poor match - for comparison)"

            # Add instruction about negative sample
            negative_instruction = ""
            if negative_sample_text:
                negative_instruction = " Do NOT select UNLIKELY - it shows what a poor match looks like."

            # Use CoT prompt when enabled (simplified for local LLM)
            if use_cot:
                prompt = f"""Match this voiceover to the best candidate using structured reasoning:{negative_instruction}

VOICEOVER: "{vo_text_clean}"

CANDIDATES:
{candidates_text}{negative_sample_text}

RUBRIC: visual_relevance (30%), topic_match (40%), keyword_overlap (20%), flow (10%)

Respond with ONLY valid JSON:
{{"selected": 1, "voiceover_themes": ["theme1"], "video_elements": ["element1"], "rubric_scores": {{"visual_relevance": 0.8, "topic_match": 0.9, "keyword_overlap": 0.7, "flow": 0.8}}, "confidence": 0.85, "reason": "topic match"}}"""
            else:
                prompt = f"""Match this voiceover to the best candidate:{negative_instruction}

VOICEOVER: "{vo_text_clean}"

CANDIDATES:
{candidates_text}{negative_sample_text}

Respond with ONLY valid JSON, no other text: {{"selected": 1, "confidence": 0.85, "reason": "topic match"}}"""

            try:
                # Call unified LLM client
                request = LLMRequest(
                    prompt=prompt,
                    response_format=ResponseFormat.JSON,
                    cache_key_prefix="matching_cot" if use_cot else "matching",
                    timeout=60
                )

                response = self.client.generate(request)

                # Log LLM API call metrics (US-159-008)
                # Handle both real responses and mock objects in tests
                duration_ms = response.request_time_ms
                tokens_used = response.tokens_used
                provider_name = response.provider or "ollama"
                model_name = response.model or self.client.model if hasattr(self.client, 'model') and not isinstance(self.client.model, type(None)) else "local"
                cached = response.cached

                # Only log if values are numeric (not mock objects)
                if isinstance(duration_ms, (int, float)) and isinstance(tokens_used, (int, type(None))):
                    if tokens_used:
                        logger.info(
                            f"LLM API call: provider={provider_name}, model={model_name}, "
                            f"duration={duration_ms:.0f}ms, tokens={tokens_used}, cached={cached}"
                        )
                    else:
                        logger.info(
                            f"LLM API call: provider={provider_name}, model={model_name}, "
                            f"duration={duration_ms:.0f}ms, cached={cached}"
                        )

                # Process result
                if response.parsed_data and isinstance(response.parsed_data, dict):
                    data = response.parsed_data
                    selected_idx = data.get('selected', 1) - 1
                    selected_idx = max(0, min(selected_idx, len(candidates) - 1))
                    confidence = max(0.0, min(1.0, float(data.get('confidence', 0.5))))

                    # Get the selected candidate for logging
                    selected_source = candidates[selected_idx][0].source_file if selected_idx < len(candidates) else "unknown"

                    # Parse CoT reasoning if CoT is enabled
                    cot_reasoning = None
                    if use_cot:
                        cot_reasoning = parse_cot_reasoning(data)
                        if cot_reasoning.is_complete:
                            weighted_score = cot_reasoning.compute_weighted_score()
                            confidence = 0.7 * weighted_score + 0.3 * confidence

                    # Log reranking decision (US-159-008)
                    reason = str(data.get('reason', 'local match'))[:50]
                    logger.info(
                        f"Rerank decision: vo_idx={idx}, selected_idx={selected_idx}, "
                        f"source={Path(selected_source).stem[:20]}, confidence={confidence:.3f}, "
                        f"reason=\"{reason}\""
                    )

                    outputs.append((
                        selected_idx,
                        confidence,
                        reason,
                        cot_reasoning
                    ))
                else:
                    logger.warning(f"Rerank fallback: vo_idx={idx}, using embedding similarity")
                    outputs.append((0, candidates[0][1] if candidates else 0.5, "local fallback", None))

            except Exception as e:
                logger.warning(f"Local LLM error: {e}, using embedding fallback")
                # Fallback
                outputs.append((0, candidates[0][1] if candidates else 0.5, "local fallback", None))

        return outputs
